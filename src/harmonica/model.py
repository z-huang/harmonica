"""Harmonica: a CQT frontend, a trunk of harmonic and ResNet blocks, three heads.

Section 3.3 of the paper, piece by piece:

* a CQT at 16 kHz, three bins a semitone, 32 ms frames, spanning the 88 keys;
* a frontend of three ResNet blocks of 5x5 convolutions at ``c1`` channels,
  the first (the stem) widening from the input through a 1x1 projection;
* a 3x1 convolution of stride 3 that widens to ``c2`` and lands on one bin a
  semitone, the output's pitch axis;
* ``l`` trunk blocks, each a harmonic layer ``ReLU(x + Norm(HConv(x)))`` and a
  ResNet block of two 3x3 convolutions;
* onset, sustain, and offset heads, each a frequency-grouped LSTM and a linear
  map to one logit per pitch and frame; optionally, not in the paper, a
  velocity head of the same form.

Instance normalization throughout, as the paper uses. The paper leaves the
stem, the 1x1 projection, and the heads' widths unstated; these give 654,291
parameters for medium, against Table 1's 679K.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from harmonica.config import (
    MIN_PITCH,
    NUM_PITCHES,
    HarmonicaModelConfig,
    resolve_widths,
)

HEADS = ("onset", "sustain", "offset")
# Not in the paper: each onset's velocity / 127, with model.head.velocity.
VELOCITY = "velocity"

# What silence, and the zero padding of a short clip, reads: log10(1 + 0).
SILENCE = 0.0


def harmonic_offsets(harmonics: int, bins_per_semitone: int = 1) -> list[int]:
    """Distances from a fundamental to its harmonics 1..n and subharmonics 2..n.

    In bins, at ``bins_per_semitone``. For n = 8 at one bin a semitone this is
    the paper's fifteen: 0, +-12, +-19, +-24, +-28, +-31, +-34, +-36.
    """
    upper = [
        round(12 * bins_per_semitone * math.log2(h)) for h in range(2, harmonics + 1)
    ]
    return sorted({0, *upper, *(-o for o in upper)})


def shift_frequency(x: torch.Tensor, offset: int) -> torch.Tensor:
    """``y[..., f, :] = x[..., f + offset, :]``, zero past either edge."""
    if offset == 0:
        return x
    bins = x.shape[-2]
    if abs(offset) >= bins:
        return torch.zeros_like(x)
    if offset > 0:
        return F.pad(x[..., offset:, :], (0, 0, 0, offset))
    return F.pad(x[..., :offset, :], (0, 0, -offset, 0))


class ShiftAggregate(nn.Module):
    """The paper's harmonic convolution: shift, concatenate, one 1x1 conv.

    Each harmonic position is one ``c x c`` matrix, so the layer holds
    ``M c^2`` weights against HD-Conv's ``3 (M - 1) / 2 c^2``. The
    concatenation is built: one GEMM over all ``M c`` inputs is several times
    faster than ``M`` small mixes shifted and summed, for ``M`` times the
    activation memory.
    """

    def __init__(self, channels: int, offsets: list[int]) -> None:
        super().__init__()
        self.offsets = offsets
        self.weight = nn.Parameter(torch.empty(channels, channels * len(offsets)))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The weight's columns are offset-major, as the concatenation is.
        stacked = torch.cat([shift_frequency(x, o) for o in self.offsets], dim=1)
        weight = self.weight[:, :, None, None].to(x.dtype)
        return F.conv2d(stacked, weight, self.bias.to(x.dtype))


class HarmonicDilatedConv(nn.Module):
    """HPPNet's HD-Conv: one dilated 3x1 branch per harmonic, summed.

    Every branch carries its own centre tap, which is the redundancy the
    shift-and-aggregate form removes. Kept for Table 2's last row.

    HPPNet's branches each have a bias; these have none. The instance norm
    after the layer cancels any bias anyway, and without them the two forms
    differ by Section 3.4's 28.5% and Table 2's 39.9% (with seven biases,
    28.9% and 40.6%).
    """

    def __init__(self, channels: int, harmonics: int) -> None:
        super().__init__()
        dilations = [round(12 * math.log2(h)) for h in range(2, harmonics + 1)]
        self.branches = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(3, 1),
                padding=(d, 0),
                dilation=(d, 1),
                bias=False,
            )
            for d in dilations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum((branch(x) for branch in self.branches), torch.zeros_like(x))


class HarmonicBlock(nn.Module):
    """``ReLU(x + Norm(HConv(x)))``."""

    def __init__(self, channels: int, conv: nn.Module) -> None:
        super().__init__()
        self.conv = conv
        self.norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """Two ``k x k`` convolutions with instance norm and a residual.

    The residual is the identity, or a 1x1 convolution when the block widens.
    """

    def __init__(self, channels: int, kernel_size: int, in_channels: int | None = None):
        super().__init__()
        in_channels = channels if in_channels is None else in_channels
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.skip = (
            nn.Identity()
            if in_channels == channels
            else nn.Conv2d(in_channels, channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.conv1(x)))
        return F.relu(self.skip(x) + self.norm2(self.conv2(h)))


class FrequencyGroupedLSTM(nn.Module):
    """An LSTM over time run at every pitch with shared weights, then a logit."""

    def __init__(self, channels: int, hidden: int, bidirectional: bool) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            channels, hidden, batch_first=True, bidirectional=bidirectional
        )
        self.linear = nn.Linear(hidden * (2 if bidirectional else 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, pitches, frames = x.shape
        sequence = x.permute(0, 2, 3, 1).reshape(batch * pitches, frames, channels)
        # Under autocast the trunk hands over bf16; some PyTorch builds (2.14's
        # cu126 aarch64 wheel) then skip cuDNN for a per-timestep LSTM several
        # times slower. Autocast casts fp32 back to bf16, so the math is the same.
        hidden, _ = self.lstm(sequence.float())
        return self.linear(hidden).view(batch, pitches, frames)


class CqtFrontend(nn.Module):
    """Audio to ``log10(1 + |CQT|)``, ``[B, samples] -> [B, 1, 88 * bins, T]``.

    The compression is the one Figure 4's first panel names ("CQT log10(x+1)");
    the text gives none. The lowest bin sits ``(bins - 1) / 2`` bins below A0,
    so after the stride-3 convolution each output row is centred on its own
    key.

    nnAudio's kernels, applied as a matmul over unfolded frames rather than
    its strided ``conv1d``: the same sums, several times faster, and never
    TF32, which cuDNN would otherwise pick for that convolution. Run in float32
    whatever the autocast around it, at full matmul precision (PyTorch's
    default): low bins sum thirty thousand taps.
    """

    # Frames multiplied at once: the unfolded frames are copied, 128 KiB each.
    FRAME_BLOCK = 2048

    def __init__(self, sample_rate: int, hop_length: int, bins_per_semitone: int):
        super().__init__()
        from nnAudio.features import CQT1992v2

        half = (bins_per_semitone - 1) // 2
        a0 = 440.0 * 2 ** ((MIN_PITCH - 69) / 12)
        self.bins_per_semitone = bins_per_semitone
        self.hop_length = hop_length
        # Kept for its buffers, which checkpoints hold; its forward is unused.
        self.cqt = CQT1992v2(
            sr=sample_rate,
            hop_length=hop_length,
            fmin=a0 * 2 ** (-half / (12 * bins_per_semitone)),
            n_bins=NUM_PITCHES * bins_per_semitone,
            bins_per_octave=12 * bins_per_semitone,
            output_format="Magnitude",
            center=True,
            pad_mode="constant",
            verbose=False,
        )
        # Real then imaginary kernels with librosa's normalisation, [2 F, width].
        scale = torch.sqrt(self.cqt.lenghts).view(-1, 1)
        kernels = torch.cat(
            [self.cqt.cqt_kernels_real[:, 0], -self.cqt.cqt_kernels_imag[:, 0]]
        ) * scale.repeat(2, 1)
        self.register_buffer("kernels", kernels, persistent=False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=audio.device.type, enabled=False):
            width = self.kernels.shape[-1]
            # nnAudio's centring: zeros of half the kernel width either side.
            padded = F.pad(audio.float(), (width // 2, width // 2))
            frames = padded.unfold(-1, width, self.hop_length)  # [B, T, width]
            block = max(1, self.FRAME_BLOCK // frames.shape[0])
            products = torch.cat(
                [
                    frames[:, first : first + block] @ self.kernels.T
                    for first in range(0, frames.shape[1], block)
                ],
                dim=1,
            )
            real, imag = products.chunk(2, dim=-1)
            magnitude = torch.hypot(real, imag).transpose(1, 2)
            return torch.log10(1 + magnitude).unsqueeze(1)


def num_frames(samples: int, hop_length: int) -> int:
    """Frames the centred CQT gives ``samples`` of audio."""
    return samples // hop_length + 1


class HarmonicaModel(nn.Module):
    """Audio (or its CQT) to onset, sustain, and offset logits ``[B, 88, T]``.

    And velocity logits, with ``config.head.velocity``.
    """

    def __init__(self, config: HarmonicaModelConfig) -> None:
        super().__init__()
        self.config = config
        widths = resolve_widths(config)
        harmonic = config.harmonic
        c1, c2 = widths.frontend_channels, widths.channels
        bins = config.cqt.bins_per_semitone

        self.cqt = CqtFrontend(config.cqt.sample_rate, config.cqt.hop_length, bins)

        # Basic Pitch's harmonic stacking: shifted copies of the CQT as input
        # channels, at its harmonics 0.5, 1, 2, ..., 7, and no harmonic layer.
        self.stack_offsets: list[int] | None = None
        in_channels = 1
        if harmonic.kind == "stacking":
            self.stack_offsets = [
                round(12 * bins * math.log2(h))
                for h in (0.5, *range(1, harmonic.harmonics))
            ]
            in_channels = len(self.stack_offsets)

        self.stem = ResBlock(c1, 5, in_channels)
        self.frontend = nn.Sequential(ResBlock(c1, 5), ResBlock(c1, 5))
        self.down = nn.Conv2d(c1, c2, kernel_size=(bins, 1), stride=(bins, 1))

        blocks: list[nn.Module] = []
        for index in range(widths.layers):
            harmonic_here = harmonic.kind in ("shift_aggregate", "hdconv") and (
                harmonic.depth == "multi" or index == 0
            )
            if harmonic_here:
                conv: nn.Module = (
                    ShiftAggregate(c2, harmonic_offsets(harmonic.harmonics))
                    if harmonic.kind == "shift_aggregate"
                    else HarmonicDilatedConv(c2, harmonic.harmonics)
                )
                blocks.append(HarmonicBlock(c2, conv))
            blocks.append(ResBlock(c2, 3))
        self.trunk = nn.Sequential(*blocks)

        self.heads = nn.ModuleDict(
            {
                name: FrequencyGroupedLSTM(
                    c2, widths.head_hidden, config.head.bidirectional
                )
                for name in (*HEADS, *((VELOCITY,) if config.head.velocity else ()))
            }
        )

    def spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """``[B, samples] -> [B, 1, F, T]`` log-CQT."""
        return self.cqt(audio)

    def forward_spectrogram(self, cqt: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits from a log-CQT the caller already computed."""
        x = cqt
        if self.stack_offsets is not None:
            x = torch.cat([shift_frequency(x, o) for o in self.stack_offsets], dim=1)
        x = self.trunk(self.down(self.frontend(self.stem(x))))
        return {name: head(x) for name, head in self.heads.items()}

    def forward(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_spectrogram(self.spectrogram(audio))


def count_parameters(model: nn.Module) -> int:
    """Trainable parameters; the CQT's kernels are buffers and not counted."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def harmonic_layer_parameters(model: HarmonicaModel) -> int:
    """Parameters in the harmonic convolutions alone, as Table 2 compares them."""
    return sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, (ShiftAggregate, HarmonicDilatedConv))
        for p in module.parameters()
    )
