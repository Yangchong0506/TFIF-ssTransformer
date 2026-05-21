"""
TFIF-ssTransformer: Time-Frequency Image Fusion with Semi-Supervised Transformer
==============================================================================
Core model architecture for multimodal FeO (ferrous oxide) soft sensing in
sintering processes. Fuses time-series process parameters with thermal images
via a Transformer encoder + MobileViT backbone, with cosine similarity
alignment and entropy regularization for semi-supervised learning.

Dependencies: PyTorch, numpy, pandas, sklearn, opencv-python, matplotlib, seaborn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_pos, DataEmbedding_pos_out
from layers.Transformer_EncDec import (
    Encoder, Encoder_mobile, EncoderLayer, EncoderLayer_mobile,
    Decoder, DecoderLayer, my_Layernorm, series_decomp, series_decomp_pri
)
from layers.SelfAttention_Family import FullAttention
from model_config import get_config
import math
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, Dict
from torch import Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Utility Functions
# ===========================================================================

def ZscoreNormalization(x):
    """Z-score normalization"""
    x = x.detach().cpu().numpy()
    mean = np.mean(x, axis=1).reshape(-1, 1)
    std = np.std(x, axis=1).reshape(-1, 1)
    x = (x - mean) / std
    return torch.FloatTensor(x).to(device)


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """Ensures all layers have a channel number divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def mtx_similar1(arr1, arr2):
    """Cosine similarity between two matrices (flattened)."""
    a1 = arr1.view(arr1.size(0), -1).detach().cpu().numpy()
    a2 = arr2.view(arr2.size(0), -1).detach().cpu().numpy()
    numer = np.sum(a1 * a2, axis=1, keepdims=True)
    denom = np.sqrt(
        np.sum(a1 ** 2, axis=1, keepdims=True) *
        np.sum(a2 ** 2, axis=1, keepdims=True)
    )
    similar = numer / denom
    similar_cos = (similar + 1) / 2
    return torch.FloatTensor(similar_cos).to(device)


# ===========================================================================
# Standard Attention Layer (replaces AutoCorrelationLayer)
# ===========================================================================

class AttentionLayer(nn.Module):
    """
    Standard multi-head scaled dot-product attention layer.
    Replaces AutoCorrelationLayer: uses FullAttention instead of
    frequency-domain auto-correlation.
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class AttentionLayer_mobile(nn.Module):
    """
    Standard multi-head attention layer for MobileViT internal blocks.
    Replaces AutoCorrelationLayer_mobile.
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer_mobile, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


# ===========================================================================
# MobileViT Building Blocks
# ===========================================================================

class ConvLayer(nn.Module):
    """2D convolution with optional normalization and activation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)
        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )
        block = nn.Sequential()
        conv_layer = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, groups=groups,
            padding=padding, bias=bias
        )
        block.add_module(name="conv", module=conv_layer)
        if use_norm:
            block.add_module(name="norm",
                module=nn.BatchNorm2d(num_features=out_channels, momentum=0.1))
        if use_act:
            block.add_module(name="act", module=nn.SiLU())
        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    """Inverted residual block from MobileNetv2."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        skip_connection: Optional[bool] = True,
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        super().__init__()
        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(name="exp_1x1",
                module=ConvLayer(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1))
        block.add_module(name="conv_3x3",
            module=ConvLayer(in_channels=hidden_dim, out_channels=hidden_dim,
                             stride=stride, kernel_size=3, groups=hidden_dim))
        block.add_module(name="red_1x1",
            module=ConvLayer(in_channels=hidden_dim, out_channels=out_channels,
                             kernel_size=1, use_act=False, use_norm=True))
        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer."""
    def __init__(self, channel, reduction=4) -> None:
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = ZscoreNormalization(y)
        y = self.fc(y).view(b, c, 1, 1)
        a = x * y.expand_as(x)
        b_out = torch.sum(a, dim=1, keepdim=False)
        return a, b_out


# ===========================================================================
# MobileViT Block (with standard Transformer attention)
# ===========================================================================

class MobileViTBlock(nn.Module):
    """
    MobileViT block with standard Transformer self-attention.
    Uses FullAttention instead of AutoCorrelation for learning global
    representations from image patches.
    """
    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int = 2,
        head_dim: int = 32,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        patch_h: int = 8,
        patch_w: int = 8,
        conv_ksize: Optional[int] = 3,
        se_in_channel: int = 64,
        factor: int = 2,
        output_atten: bool = True,
        moving_avg: int = 5,
        *args, **kwargs
    ) -> None:
        super().__init__()

        conv_3x3_in = ConvLayer(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels, out_channels=transformer_dim,
            kernel_size=1, stride=1, use_norm=False, use_act=False
        )
        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim, out_channels=in_channels,
            kernel_size=1, stride=1
        )
        conv_3x3_out = ConvLayer(
            in_channels=2 * in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1
        )

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        d_model = transformer_dim
        n_heads = d_model // head_dim
        d_ff = d_model * 4

        # Standard Transformer attention replaces AutoCorrelation
        self.global_rep = Encoder_mobile([
            EncoderLayer_mobile(
                AttentionLayer_mobile(
                    FullAttention(False, factor,
                                  attention_dropout=dropout,
                                  output_attention=False),
                    d_model, n_heads),
                d_model, d_ff,
                moving_avg=moving_avg,
                dropout=dropout,
                activation="gelu"
            ) for _ in range(n_transformer_blocks)
        ], norm_layer=my_Layernorm(d_model))

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize
        self.se = SELayer(se_in_channel, 8)

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear",
                              align_corners=False)
            interpolate = True

        num_patch_w = new_w // patch_w
        num_patch_h = new_h // patch_h
        num_patches = num_patch_h * num_patch_w

        x = x.reshape(batch_size * in_channels * num_patch_h,
                      patch_h, num_patch_w, patch_w)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        x = x.transpose(1, 3)
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }
        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tuple[Tensor, Tensor]:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC."

        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area,
            info_dict["total_patches"], -1
        )

        x, xx = self.se(x)
        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        x = x.transpose(1, 3)
        x = x.reshape(batch_size * channels * num_patch_h,
                      num_patch_w, self.patch_h, self.patch_w)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, channels,
                      num_patch_h * self.patch_h,
                      num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(x, size=info_dict["orig_size"],
                              mode="bilinear", align_corners=False)
        return x, xx

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        res = x
        fm = self.local_rep(x)

        patches, info_dict = self.unfolding(fm)
        patches, atten = self.global_rep(patches)

        fm_att, fm_seq = self.folding(x=patches, info_dict=info_dict)
        fm = self.conv_proj(fm_att)
        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm, fm_seq


# ===========================================================================
# MobileViT Inner Architecture
# ===========================================================================

class MobileViT_inner(nn.Module):
    """MobileViT architecture for image feature extraction."""
    def __init__(self, model_cfg: Dict, num_classes: int = 1000,
                 feature_dim: int = 32):
        super().__init__()
        image_channels = 1
        out_channels = 4

        self.conv_1 = ConvLayer(
            in_channels=image_channels, out_channels=out_channels,
            kernel_size=3, stride=2
        )
        self.layer_1, out_channels = self._make_layer(
            input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(
            input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer_3, out_channels = self._make_layer(
            input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(
            input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(
            input_channel=out_channels, cfg=model_cfg["layer5"])

        exp_channels = min(
            model_cfg["last_layer_exp_factor"] * out_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            in_channels=out_channels, out_channels=exp_channels,
            kernel_size=1
        )

        self.classifier = nn.Sequential()
        self.classifier.add_module(
            name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module(name="flatten", module=nn.Flatten())
        if 0.0 < model_cfg["cls_dropout"] < 1.0:
            self.classifier.add_module(
                name="dropout",
                module=nn.Dropout(p=model_cfg["cls_dropout"]))
        self.classifier.add_module(
            name="fc",
            module=nn.Linear(in_features=exp_channels,
                             out_features=num_classes))

        self.apply(self.init_parameters)
        self.norm = my_Layernorm(feature_dim)
        self.layernorm = nn.LayerNorm(feature_dim)

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(
                input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict
                              ) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 2)
        block = []
        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict
                        ) -> Tuple[nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 2)
            )
            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError(
                "Transformer input dimension should be divisible by "
                "head dimension.")

        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 8),
            patch_w=cfg.get("patch_w", 20),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3,
            se_in_channel=cfg.get("se_in_channel", 320),
            factor=2,
            output_atten=False,
            moving_avg=5,
        ))
        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x, _ = self.layer_3(x)
        x, _ = self.layer_4(x)
        x, x_seq = self.layer_5(x)
        x_seq = self.norm(x_seq)
        return x_seq


# ===========================================================================
# MobileViT Factory Functions
# ===========================================================================

def mobile_vit_xxx_small(num_classes=1000, feature_dim=16):
    config = get_config("xxx_small")
    return MobileViT_inner(config, num_classes=num_classes,
                           feature_dim=feature_dim)


def mobile_vit_xx_small(num_classes=1000, feature_dim=32):
    config = get_config("xx_small")
    return MobileViT_inner(config, num_classes=num_classes,
                           feature_dim=feature_dim)


def mobile_vit_x_small(num_classes=1000, feature_dim=32):
    config = get_config("x_small")
    return MobileViT_inner(config, num_classes=num_classes,
                           feature_dim=feature_dim)


def mobile_vit_small(num_classes=1000, feature_dim=32):
    config = get_config("small")
    return MobileViT_inner(config, num_classes=num_classes,
                           feature_dim=feature_dim)


# ===========================================================================
# Positional Encoding
# ===========================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


# ===========================================================================
# TFIF-ssTransformer Core Model
# ===========================================================================

class Transformer_Model_cos_sim(nn.Module):
    """
    TFIF-ssTransformer: Multimodal semi-supervised Transformer for FeO
    soft sensing in sintering processes.

    Architecture:
        - Time-series branch: Transformer encoder with series decomposition
        - Image branch: MobileViT for thermal image feature extraction
        - Cosine similarity alignment between modalities
        - Entropy regularization for semi-supervised learning

    Uses standard scaled dot-product attention (FullAttention) instead of
    frequency-domain auto-correlation.
    """
    def __init__(self, enc_in, seq_len, label_len, pred_len, moving_avg,
                 factor, dropout, output_atten, d_model, n_heads, d_ff,
                 activation, e_layers, d_layers, c_out):
        super(Transformer_Model_cos_sim, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_atten

        # Series decomposition
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)
        self.decomp_pri = series_decomp_pri()
        self.DataEmbedding_pos = DataEmbedding_pos(enc_in, d_model, dropout)
        self.DataEmbedding_pos_out = DataEmbedding_pos_out(
            1, d_model, dropout)

        # Encoder with standard Transformer attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor,
                                      attention_dropout=dropout,
                                      output_attention=output_atten),
                        d_model, n_heads),
                    d_model, d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )

        # Decoder with standard Transformer attention
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, factor,
                                      attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor,
                                      attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    d_model, c_out, d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

        self.out_fc = nn.Linear(seq_len * c_out, c_out)
        self.mobilevit = mobile_vit_xx_small(num_classes=3, feature_dim=32)
        self.out_reduce_dim = nn.Linear(d_model + 32, d_model)
        self.mobile_linear = nn.Linear(32, 6)
        self.auto_linear = nn.Linear(d_model, 6)
        self.mobile_linear_inverse = nn.Linear(6, 32)
        self.auto_linear_inverse = nn.Linear(6, d_model)
        self.eudistance = nn.PairwiseDistance(p=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x_pic, if_save_dynamic=False, step=0, epoch=0,
                beta=0, enc_self_mask=None, dec_self_mask=None,
                dec_enc_mask=None):
        # Series decomposition initialization
        x_enc = x[:, :, :-2]
        x_dec = x[:, :, [-2]]
        seasonal_init, trend_init = self.decomp(x_dec)

        # Encoder: time-series branch + image branch
        enc_out = self.DataEmbedding_pos(x_enc)
        enc_out_auto, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out_mobile = self.mobilevit(x_pic)

        # Dimensionality reduction
        enc_out_mobile_reduce = self.mobile_linear(enc_out_mobile)
        enc_out_auto_reduce = self.auto_linear(enc_out_auto)

        # Cosine similarity alignment between modalities
        cos_sim = torch.abs(torch.cosine_similarity(
            enc_out_auto_reduce, enc_out_mobile_reduce, dim=1))
        log_prob = F.log_softmax(cos_sim, dim=-1)
        log_prob = log_prob.squeeze(-1)
        prob = torch.exp(log_prob)
        entropy1 = -torch.sum(prob * log_prob, dim=-1)
        cos_sim = torch.mean(cos_sim, dim=-1)
        cos_sim_final = 1 - cos_sim
        entropy = entropy1

        # Inverse projection + residual connection
        enc_out_mobile_inverse = self.mobile_linear_inverse(
            enc_out_mobile_reduce)
        enc_out_auto_inverse = self.auto_linear_inverse(
            enc_out_auto_reduce)
        enc_out_mobile = enc_out_mobile_inverse + enc_out_mobile
        enc_out_auto = enc_out_auto_inverse + enc_out_auto

        # Multimodal fusion
        enc_out_cat = torch.cat((enc_out_auto, enc_out_mobile), 2)
        enc_out = self.out_reduce_dim(enc_out_cat)

        # Decoder
        dec_out = seasonal_init
        dec_out = self.DataEmbedding_pos_out(dec_out)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out,
            x_mask=dec_self_mask, cross_mask=dec_enc_mask,
            trend=trend_init)

        # Final output
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            results = self.out_fc(dec_out.flatten(start_dim=1))
            return results, cos_sim_final, entropy, dec_out


