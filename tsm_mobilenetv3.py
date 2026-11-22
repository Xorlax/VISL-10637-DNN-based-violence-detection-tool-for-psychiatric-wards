"""
TSM + MobileNetV3 building blocks for video violence detection (PyTorch).

Sources this file is adapted from:
  * MobileNetV3 — torchvision's reference implementation
      https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
      Paper: Howard et al., "Searching for MobileNetV3", ICCV 2019 (arXiv:1905.02244)
  * Temporal Shift Module — official MIT HAN Lab code (MIT license)
      https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/temporal_shift.py
      Paper: Lin, Gan, Han, "TSM: Temporal Shift Module for Efficient Video Understanding",
             ICCV 2019 (arXiv:1811.08383)
  * Squeeze-and-Excitation block (used inside MobileNetV3, implemented as in torchvision)
      Paper: Hu, Shen, Sun, "Squeeze-and-Excitation Networks", CVPR 2018 (arXiv:1709.01507)

Layer/attribute names deliberately mirror torchvision, so ImageNet-pretrained
weights from torchvision can be loaded directly (see `load_imagenet_weights`).
The temporal shift adds zero parameters, so it does not break that compatibility.

Tensor shape convention for video:  (B, T, C, H, W)
    B = batch size (clips), T = frames per clip (n_segment), C = 3 (RGB)
Inside the 2D backbone, frames are flattened into the batch dimension:
    (N, C, H, W) with N = B*T, frames of the same clip stored contiguously.
All tensors are float (float32 by default, or float16 under autocast).
"""

from functools import partial
from typing import Callable, List, Optional

import torch  # type: ignore[reportMissingImports]
from torch import nn, Tensor  # type: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Round a channel count to the nearest multiple of `divisor` (hardware friendly).

    Taken from the original TF MobileNet repo, as used in torchvision.
    Guarantees the result is never more than 10% smaller than `v`.

    Args:
        v (float): Desired number of channels (may be fractional, e.g. after width_mult).
        divisor (int): The result will be a multiple of this. Default 8.
        min_value (int, optional): Lower bound on the result. Defaults to `divisor`.

    Returns:
        int: Rounded channel count, a multiple of `divisor` and >= `min_value`.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # make sure rounding down doesn't drop more than 10%
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    """Conv2d -> BatchNorm2d -> activation (optional).

    Same structure (and state_dict keys 0/1) as torchvision's Conv2dNormActivation.
    Padding is chosen as "same" padding, so the spatial size only changes by `stride`.

    Args:
        in_planes (int): Number of input channels C_in.
        out_planes (int): Number of output channels C_out.
        kernel_size (int): Square kernel size k. Default 3.
        stride (int): Convolution stride (1 or 2 in this model). Default 1.
        groups (int): Conv groups; set to C_in for a depthwise conv. Default 1.
        dilation (int): Conv dilation. Default 1.
        norm_layer (callable, optional): Normalization layer factory, called as
            norm_layer(out_planes). Defaults to nn.BatchNorm2d.
        activation_layer (callable, optional): Activation factory, called as
            activation_layer(inplace=True). None = no activation. Default nn.ReLU.

    Input (forward):
        Tensor of shape (N, C_in, H, W), float.
    Output (forward):
        Tensor of shape (N, C_out, ceil(H/stride), ceil(W/stride)), float.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ):
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        layers: List[nn.Module] = [
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                      dilation=dilation, groups=groups, bias=False),
            norm_layer(out_planes),
        ]
        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))
        super().__init__(*layers)
        self.out_channels = out_planes


# ---------------------------------------------------------------------------
# MobileNetV3 building blocks
# ---------------------------------------------------------------------------
class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block (Hu et al., CVPR 2018) as used in MobileNetV3.

    Source: torchvision.ops.SqueezeExcitation, with Hardsigmoid as in torchvision's MobileNetV3.

    global-avg-pool -> 1x1 conv (reduce) -> ReLU -> 1x1 conv (expand) -> Hardsigmoid -> scale.
    Learns a per-channel weight in [0, 1] and multiplies the input by it
    (channel attention). Shape is unchanged.

    Args:
        input_channels (int): Number of channels C of the input feature map.
        squeeze_channels (int): Size of the bottleneck (reduced) layer.
    """

    def __init__(self, input_channels: int, squeeze_channels: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = nn.ReLU()
        self.scale_activation = nn.Hardsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Re-weight the channels of `x`.

        Args:
            x (Tensor): Feature map, shape (N, C, H, W), float.

        Returns:
            Tensor: Same shape as `x`, (N, C, H, W), each channel scaled by a value in [0, 1].
        """
        scale = self.avgpool(x)                         # (N, C, 1, 1)
        scale = self.activation(self.fc1(scale))        # (N, squeeze, 1, 1)
        scale = self.scale_activation(self.fc2(scale))  # (N, C, 1, 1)
        return scale * x


class InvertedResidualConfig:
    """Configuration of one bottleneck block = one row of Table 1 / Table 2 of the MobileNetV3 paper.

    Plain data holder (no forward pass). Channel counts are scaled by
    `width_mult` and rounded to a multiple of 8.
    Source: torchvision's mobilenetv3.py (InvertedResidualConfig).

    Args:
        input_channels (int): Channels entering the block (before width_mult).
        kernel (int): Depthwise kernel size (3 or 5).
        expanded_channels (int): Channels after the 1x1 expansion ("exp size").
        out_channels (int): Channels leaving the block.
        use_se (bool): Whether to add a Squeeze-and-Excitation layer.
        activation (str): "HS" = Hardswish, "RE" = ReLU.
        stride (int): Depthwise stride, 1 or 2.
        dilation (int): Depthwise dilation. Default 1.
        width_mult (float): Channel multiplier for the whole network. Default 1.0.
    """

    def __init__(self, input_channels: int, kernel: int, expanded_channels: int,
                 out_channels: int, use_se: bool, activation: str, stride: int,
                 dilation: int = 1, width_mult: float = 1.0):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # "HS" = Hardswish, "RE" = ReLU
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float) -> int:
        """Scale a channel count by `width_mult` and round it to a multiple of 8.

        Args:
            channels (int): Base channel count from the paper's table.
            width_mult (float): Network width multiplier.

        Returns:
            int: Adjusted channel count (multiple of 8).
        """
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    """MobileNetV3 bottleneck ("bneck"):
        1x1 expand -> kxk depthwise -> [SE] -> 1x1 project (linear) [+ residual]

    The residual (skip) connection is used only when stride == 1 and
    input_channels == out_channels.

    If `temporal_shift` is given, it is applied at the start of the residual
    branch only (TSM's "residual shift"), so the identity path keeps the
    un-shifted, current-frame features.

    Source: torchvision's mobilenetv3.py (InvertedResidual); the temporal shift
    placement follows the TSM paper (Sec. 3.3, residual shift).

    Args:
        cnf (InvertedResidualConfig): Block configuration (channels, kernel, SE, activation, stride).
        norm_layer (callable): Normalization layer factory, e.g. partial(nn.BatchNorm2d, ...).
        temporal_shift (nn.Module, optional): A TemporalShift module, or None for a
            plain per-frame 2D block. Default None.

    Raises:
        ValueError: If cnf.stride is not 1 or 2.
    """

    def __init__(self, cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module],
                 temporal_shift: Optional[nn.Module] = None):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError(f"illegal stride value: expected 1 or 2, got {cnf.stride}")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        layers: List[nn.Module] = []
        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer))
        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels,
                                       kernel_size=cnf.kernel, stride=stride, dilation=cnf.dilation,
                                       groups=cnf.expanded_channels,
                                       norm_layer=norm_layer, activation_layer=activation_layer))
        # squeeze-and-excitation
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(SqueezeExcitation(cnf.expanded_channels, squeeze_channels))
        # project (no activation -> "linear bottleneck")
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self.temporal_shift = temporal_shift  # parameter-free; None = plain 2D block

    def forward(self, x: Tensor) -> Tensor:
        """Run the bottleneck block.

        Args:
            x (Tensor): Shape (N, C_in, H, W), float. When a temporal shift is used,
                N = B*T and the T frames of each clip must be contiguous.

        Returns:
            Tensor: Shape (N, C_out, ceil(H/stride), ceil(W/stride)), float.
        """
        branch_in = self.temporal_shift(x) if self.temporal_shift is not None else x
        result = self.block(branch_in)
        if self.use_res_connect:
            result = result + x
        return result


# ---------------------------------------------------------------------------
# Temporal Shift Module (TSM)
# ---------------------------------------------------------------------------
def temporal_shift(x: Tensor, n_segment: int, fold_div: int = 8) -> Tensor:
    """Bidirectional temporal shift: move a fraction of channels along time. Zero params, zero FLOPs.

    Of the C channels:
        first  C/fold_div : take features from frame t+1 (shift "left" / backward in time)
        second C/fold_div : take features from frame t-1 (shift "right" / forward in time)
        rest              : unchanged
    Out-of-range positions (first / last frame) are zero-padded.
    Source: TemporalShift.shift in the official TSM code (ops/temporal_shift.py).

    NOTE for the real-time version: the bidirectional shift needs the *next*
    frame. For online/causal inference use `causal=True` in TemporalShift
    (uni-directional shift, past frames only), as in TSM's online demo.

    Args:
        x (Tensor): Shape (B*T, C, H, W), float. The T frames of each clip must be
            contiguous in the first dimension; B*T must be divisible by n_segment.
        n_segment (int): T, number of frames per clip.
        fold_div (int): 1/fold_div of the channels is shifted in each direction. Default 8.

    Returns:
        Tensor: New tensor, same shape and dtype as `x`, (B*T, C, H, W).

    Raises:
        AssertionError: If B*T is not divisible by n_segment.
    """
    nt, c, h, w = x.size()
    assert nt % n_segment == 0, f"B*T={nt} is not divisible by n_segment={n_segment}"
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)   # (B, T, C, H, W)
    fold = c // fold_div

    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]                   # shift left  (future -> now)
    out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]   # shift right (past -> now)
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]              # not shifted
    return out.view(nt, c, h, w)   # (B*T, C, H, W)


def causal_temporal_shift(x: Tensor, n_segment: int, fold_div: int = 8) -> Tensor:
    """Uni-directional temporal shift: only past -> present, so it can run in real time.

    The first C/fold_div channels of frame t take their values from frame t-1
    (zero for the first frame); all other channels are unchanged.
    Source: the uni-directional (online) shift of the official TSM code's online demo.

    Args:
        x (Tensor): Shape (B*T, C, H, W), float, frames of each clip contiguous;
            B*T must be divisible by n_segment.
        n_segment (int): T, number of frames per clip.
        fold_div (int): 1/fold_div of the channels is shifted. Default 8.

    Returns:
        Tensor: New tensor, same shape and dtype as `x`, (B*T, C, H, W).

    Raises:
        AssertionError: If B*T is not divisible by n_segment.
    """
    nt, c, h, w = x.size()
    assert nt % n_segment == 0, f"B*T={nt} is not divisible by n_segment={n_segment}"
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)   # (B, T, C, H, W)
    fold = c // fold_div

    out = torch.zeros_like(x)
    out[:, 1:, :fold] = x[:, :-1, :fold]   # past -> now
    out[:, :, fold:] = x[:, :, fold:]
    return out.view(nt, c, h, w)   # (B*T, C, H, W)


class TemporalShift(nn.Module):
    """nn.Module wrapper around the shift functions, so the shift can be dropped into any block.

    Has no learnable parameters.

    Args:
        n_segment (int): T, number of frames per clip. Default 8.
        fold_div (int): 1/fold_div of the channels is shifted per direction. Default 8.
        causal (bool): True = past-only shift (`causal_temporal_shift`, for real time);
            False = bidirectional shift (`temporal_shift`). Default False.
    """

    def __init__(self, n_segment: int = 8, fold_div: int = 8, causal: bool = False):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div
        self.causal = causal

    def forward(self, x: Tensor) -> Tensor:
        """Apply the temporal shift.

        Args:
            x (Tensor): Shape (B*T, C, H, W), float, frames of each clip contiguous.

        Returns:
            Tensor: Same shape as `x`, (B*T, C, H, W).
        """
        fn = causal_temporal_shift if self.causal else temporal_shift
        return fn(x, self.n_segment, self.fold_div)

    def extra_repr(self) -> str:
        """Text shown inside print(model) for this module.

        Returns:
            str: e.g. "n_segment=8, fold_div=8, causal=False".
        """
        return f"n_segment={self.n_segment}, fold_div={self.fold_div}, causal={self.causal}"


# ---------------------------------------------------------------------------
# Architecture tables (MobileNetV3 paper, Tables 1 & 2)
# ---------------------------------------------------------------------------
def mobilenet_v3_conf(arch: str = "large", width_mult: float = 1.0):
    """Build the list of bottleneck configs for MobileNetV3-Large or -Small.

    Source: torchvision's mobilenetv3.py (_mobilenet_v3_conf).

    Args:
        arch (str): "large" or "small". Default "large".
        width_mult (float): Channel multiplier applied to every layer. Default 1.0.

    Returns:
        tuple:
            setting (List[InvertedResidualConfig]): One config per bottleneck,
                15 entries for "large", 11 for "small".
            last_channel (int): Hidden size of the classifier head
                (1280 for "large", 1024 for "small", at width_mult=1.0).

    Raises:
        ValueError: If `arch` is not "large" or "small".
    """
    bneck = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "large":
        setting = [
            #     in  k  exp  out   SE     act  s
            bneck(16, 3, 16, 16, False, "RE", 1),
            bneck(16, 3, 64, 24, False, "RE", 2),   # C1
            bneck(24, 3, 72, 24, False, "RE", 1),
            bneck(24, 5, 72, 40, True, "RE", 2),    # C2
            bneck(40, 5, 120, 40, True, "RE", 1),
            bneck(40, 5, 120, 40, True, "RE", 1),
            bneck(40, 3, 240, 80, False, "HS", 2),  # C3
            bneck(80, 3, 200, 80, False, "HS", 1),
            bneck(80, 3, 184, 80, False, "HS", 1),
            bneck(80, 3, 184, 80, False, "HS", 1),
            bneck(80, 3, 480, 112, True, "HS", 1),
            bneck(112, 3, 672, 112, True, "HS", 1),
            bneck(112, 5, 672, 160, True, "HS", 2),  # C4
            bneck(160, 5, 960, 160, True, "HS", 1),
            bneck(160, 5, 960, 160, True, "HS", 1),
        ]
        last_channel = adjust(1280)
    elif arch == "small":
        setting = [
            bneck(16, 3, 16, 16, True, "RE", 2),    # C1
            bneck(16, 3, 72, 24, False, "RE", 2),   # C2
            bneck(24, 3, 88, 24, False, "RE", 1),
            bneck(24, 5, 96, 40, True, "HS", 2),    # C3
            bneck(40, 5, 240, 40, True, "HS", 1),
            bneck(40, 5, 240, 40, True, "HS", 1),
            bneck(40, 5, 120, 48, True, "HS", 1),
            bneck(48, 5, 144, 48, True, "HS", 1),
            bneck(48, 5, 288, 96, True, "HS", 2),   # C4
            bneck(96, 5, 576, 96, True, "HS", 1),
            bneck(96, 5, 576, 96, True, "HS", 1),
        ]
        last_channel = adjust(1024)
    else:
        raise ValueError(f"Unsupported arch {arch!r} (use 'large' or 'small')")
    return setting, last_channel


# ---------------------------------------------------------------------------
# MobileNetV3 (2D, per-frame) with optional TSM in its residual blocks
# ---------------------------------------------------------------------------
class MobileNetV3(nn.Module):
    """MobileNetV3 image classifier (works on single frames).

    Pass `n_segment` to insert TSM into every residual (stride-1, same-channel)
    bottleneck — the placement TSM uses for MobileNetV2.
    Total downsampling of the feature extractor is 32x (e.g. 224 -> 7).
    Source: torchvision's mobilenetv3.py (MobileNetV3), plus the TSM insertion.

    Args:
        inverted_residual_setting (List[InvertedResidualConfig]): Bottleneck configs,
            usually from `mobilenet_v3_conf`.
        last_channel (int): Hidden size of the classifier head (1280 large / 1024 small).
        num_classes (int): Number of output classes K. Default 1000 (ImageNet).
        dropout (float): Dropout probability in the classifier head. Default 0.2.
        n_segment (int, optional): Frames per clip T. None = no temporal shift
            (plain image model). Default None.
        shift_div (int): 1/shift_div of the channels is shifted per direction. Default 8.
        causal_shift (bool): Use the past-only (real-time) shift. Default False.
    """

    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int, num_classes: int = 1000, dropout: float = 0.2,
                 n_segment: Optional[int] = None, shift_div: int = 8, causal_shift: bool = False):
        super().__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)  # torchvision's BN settings

        layers: List[nn.Module] = []
        # stem
        first_out = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(3, first_out, kernel_size=3, stride=2,
                                       norm_layer=norm_layer, activation_layer=nn.Hardswish))
        # bottlenecks
        for cnf in inverted_residual_setting:
            is_residual = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
            shift = (TemporalShift(n_segment, shift_div, causal_shift)
                     if (n_segment is not None and is_residual) else None)
            layers.append(InvertedResidual(cnf, norm_layer, temporal_shift=shift))
        # last conv
        last_in = inverted_residual_setting[-1].out_channels
        last_out = 6 * last_in  # torchvision: lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(last_in, last_out, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_out, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize all weights in place (same scheme as torchvision).

        Conv: Kaiming-normal (fan_out), bias 0. BatchNorm/GroupNorm: weight 1, bias 0.
        Linear: normal(0, 0.01), bias 0.

        Returns:
            None
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        """Extract a pooled feature vector per image/frame (no classifier head).

        Args:
            x (Tensor): Images, shape (N, 3, H, W), float, ImageNet-normalized.
                With TSM enabled, N = B*T with frames of each clip contiguous.

        Returns:
            Tensor: Shape (N, D), float, where D = 6 * last bottleneck channels
                (960 for "large", 576 for "small", at width_mult=1.0).
        """
        x = self.features(x)                       # (N, D, H/32, W/32)
        return torch.flatten(self.avgpool(x), 1)   # (N, D)

    def forward(self, x: Tensor) -> Tensor:
        """Classify each image/frame.

        Args:
            x (Tensor): Images, shape (N, 3, H, W), float, ImageNet-normalized.

        Returns:
            Tensor: Logits (before softmax), shape (N, num_classes), float.
        """
        return self.classifier(self.forward_features(x))


# ---------------------------------------------------------------------------
# Video model: TSM-MobileNetV3 for violence / non-violence
# ---------------------------------------------------------------------------
class TSMMobileNetV3(nn.Module):
    """Video classifier: MobileNetV3 + TSM, with per-frame logits averaged over time
    (TSN/TSM "avg consensus").

    Args:
        num_classes (int): Number of output classes K. Default 2 (violence / non-violence).
        n_segment (int): Frames per clip T; every input clip must have exactly T frames. Default 8.
        arch (str): "large" or "small". Default "large".
        width_mult (float): Channel multiplier. Must be 1.0 if pretrained=True. Default 1.0.
        shift_div (int): 1/shift_div of the channels is shifted per direction. Default 8.
        causal_shift (bool): Use the past-only (real-time) shift. Default False.
        dropout (float): Dropout probability in the classifier head. Default 0.5.
        pretrained (bool): Load ImageNet weights from torchvision (downloads them
            on first use). Default True.

    Input (forward):
        Tensor of shape (B, T, 3, H, W), float, ImageNet-normalized, T == n_segment.
    Output (forward):
        Tensor of shape (B, num_classes), float logits (before softmax).
    """

    def __init__(self, num_classes: int = 2, n_segment: int = 8, arch: str = "large",
                 width_mult: float = 1.0, shift_div: int = 8, causal_shift: bool = False,
                 dropout: float = 0.5, pretrained: bool = True):
        super().__init__()
        self.n_segment = n_segment
        setting, last_channel = mobilenet_v3_conf(arch, width_mult)
        self.base = MobileNetV3(setting, last_channel, num_classes=1000, dropout=dropout,
                                n_segment=n_segment, shift_div=shift_div, causal_shift=causal_shift)
        if pretrained:
            load_imagenet_weights(self.base, arch, width_mult)
        # replace the ImageNet head with our own
        in_f = self.base.classifier[-1].in_features
        self.base.classifier[-1] = nn.Linear(in_f, num_classes)
        nn.init.normal_(self.base.classifier[-1].weight, 0, 0.01)  # same init as _init_weights
        nn.init.zeros_(self.base.classifier[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a batch of video clips.

        Args:
            x (Tensor): Clips, shape (B, T, 3, H, W), float, T must equal n_segment.

        Returns:
            Tensor: Clip-level logits, shape (B, num_classes), float.

        Raises:
            AssertionError: If T != n_segment.
        """
        b, t, c, h, w = x.shape
        assert t == self.n_segment, f"expected {self.n_segment} frames, got {t}"
        x = x.reshape(b * t, c, h, w)          # frames of a clip stay contiguous
        logits = self.base(x)                   # (B*T, K)
        return logits.view(b, t, -1).mean(dim=1)  # (B, K)


def load_imagenet_weights(model: MobileNetV3, arch: str = "large", width_mult: float = 1.0):
    """Copy torchvision's ImageNet weights into our model (parameter names match 1:1).

    Downloads the weights the first time (cached by torchvision afterwards).
    Modifies `model` in place.

    Args:
        model (MobileNetV3): Model built with the same `arch`, num_classes=1000.
        arch (str): "large" or "small". Default "large".
        width_mult (float): Must be 1.0 (the only width torchvision provides). Default 1.0.

    Returns:
        MobileNetV3: The same `model` object, now holding the pretrained weights.

    Raises:
        ValueError: If `arch` is not "large"/"small", or width_mult != 1.0.
        RuntimeError: If the state_dict keys/shapes don't match (e.g. model built with another arch).
    """
    if arch not in ("large", "small"):
        raise ValueError(f"Unsupported arch {arch!r} (use 'large' or 'small')")
    if width_mult != 1.0:
        raise ValueError(f"torchvision only provides pretrained weights for width_mult=1.0, "
                         f"got {width_mult}")
    # Import lazily so torchvision remains an optional runtime dependency and
    # static analyzers do not need to resolve it when checking this module.
    from importlib import import_module
    models = import_module("torchvision.models")
    if arch == "large":
        tv = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )
    else:
        tv = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
    model.load_state_dict(tv.state_dict(), strict=True)  # TSM has no params -> exact match
    return model


# ---------------------------------------------------------------------------
# Quick self-test:  python tsm_mobilenetv3.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # tiny input: 3 frames, 8 channels; frame k holds the value k+1 in every channel
    t = torch.arange(1, 4).float().view(3, 1, 1, 1).repeat(1, 8, 1, 1)  # (3, 8, 1, 1)
    unshifted = torch.tensor([1., 2., 3.])

    # 1) bidirectional shift: channel 0 <- next frame, channel 1 <- previous frame
    s = temporal_shift(t, n_segment=3, fold_div=8).view(3, 8)  # (T, C)
    assert torch.equal(s[:, 0], torch.tensor([2., 3., 0.])), f"channel 0 wrong: {s[:, 0]}"
    assert torch.equal(s[:, 1], torch.tensor([0., 1., 2.])), f"channel 1 wrong: {s[:, 1]}"
    for ch in range(2, 8):
        assert torch.equal(s[:, ch], unshifted), f"channel {ch} should be unchanged: {s[:, ch]}"
    print("[1] temporal_shift values ... OK")

    # 2) causal shift: only channel 0 <- previous frame
    s = causal_temporal_shift(t, n_segment=3, fold_div=8).view(3, 8)  # (T, C)
    assert torch.equal(s[:, 0], torch.tensor([0., 1., 2.])), f"channel 0 wrong: {s[:, 0]}"
    for ch in range(1, 8):
        assert torch.equal(s[:, ch], unshifted), f"channel {ch} should be unchanged: {s[:, ch]}"
    print("[2] causal_temporal_shift values ... OK")

    # 3) bad frame count must fail loudly
    try:
        temporal_shift(torch.zeros(7, 8, 1, 1), n_segment=3)
        raise RuntimeError("expected an AssertionError for B*T=7, n_segment=3")
    except AssertionError:
        pass
    print("[3] non-divisible B*T rejected ... OK")

    # 4) torchvision compatibility (no download needed): keys load 1:1, and without TSM
    #    our model gives exactly torchvision's output for the same weights
    from torchvision.models import mobilenet_v3_large, mobilenet_v3_small  # type: ignore[reportMissingImports]
    x = torch.randn(2, 3, 64, 64)
    for arch, tv_fn in [("large", mobilenet_v3_large), ("small", mobilenet_v3_small)]:
        tv = tv_fn(weights=None).eval()
        setting, last = mobilenet_v3_conf(arch)
        MobileNetV3(setting, last, n_segment=8).load_state_dict(tv.state_dict(), strict=True)
        plain = MobileNetV3(setting, last).eval()
        plain.load_state_dict(tv.state_dict(), strict=True)
        with torch.no_grad():
            assert torch.allclose(plain(x), tv(x), atol=1e-5), f"[{arch}] output differs from torchvision"
    print("[4] torchvision state_dict + outputs match (large, small) ... OK")

    # 5) video model: output shape and number of inserted shifts
    #    (MobileNetV3-Large has 10 residual bottlenecks, Small has 6)
    for arch, expected_shifts in [("large", 10), ("small", 6)]:
        model = TSMMobileNetV3(num_classes=2, n_segment=8, arch=arch, pretrained=False).eval()
        clip = torch.randn(2, 8, 3, 112, 112)   # (B, T, C, H, W): 2 clips x 8 frames
        with torch.no_grad():
            out = model(clip)                   # (B, K)
        assert out.shape == (2, 2), f"[{arch}] expected logits (2, 2), got {tuple(out.shape)}"
        n_shift = sum(isinstance(m, TemporalShift) for m in model.modules())
        assert n_shift == expected_shifts, f"[{arch}] expected {expected_shifts} shifts, got {n_shift}"
    print("[5] TSMMobileNetV3 logits shape + shift count (large, small) ... OK")

    # 6) wrong number of frames must fail loudly
    try:
        model(torch.randn(1, 4, 3, 112, 112))
        raise RuntimeError("expected an AssertionError for T=4, n_segment=8")
    except AssertionError:
        pass
    print("[6] wrong T rejected ... OK")

    print("All 6 checks passed.")
