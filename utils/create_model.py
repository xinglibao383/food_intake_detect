import torch.nn as nn
import timm
from models import unet, unet_1D, segformer, SwinTransformer_1D, SwinTransformerV2_1D


def get_unet(base_c=64):
    return unet.UNet(base_c)


def get_unet_1d(base_c=64, mode="up_sample"):
    if mode in ["watch", "glasses"]:
        return unet_1D.UNet1D(base_c=base_c, in_channels=6, num_classes=6)
    elif mode in ["up_sample", "down_sample"]:
        return unet_1D.UNet1D(base_c=base_c, in_channels=12, num_classes=12)


def get_segformer():
    model = segformer.Segformer(img_size=512, in_chans=1, num_classes=1,
                                embed_dims=[32, 64, 128, 256],
                                num_heads=[1, 2, 4, 8],
                                mlp_ratios=[2, 2, 2, 2],
                                drop_rate=0.1,
                                attn_drop_rate=0.1,
                                drop_path_rate=0.1,
                                norm_layer=nn.LayerNorm,
                                depths=[2, 4, 8, 2],
                                sr_ratios=[4, 2, 1, 1],
                                decoder_dim=256)
    return model


def get_swin_transformer_1d():
    """
    net = SwinTransformer_1D.SwinTransformer_1D(
        num_classes=11,
        in_chans=12,
        embed_dim=128,
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        use_checkpoint=False)
    """
    net = SwinTransformer_1D.SwinTransformer_1D(
        num_classes=11,
        in_chans=12,
        embed_dim=32,
        depths=[1, 1, 2, 1],
        num_heads=[1, 2, 4, 8],
        window_size=7,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        use_checkpoint=False)
    return net


def get_swin_transformer_v2_1d():
    """
    net = SwinTransformerV2_1D.SwinTransformerV2_1D(seq_len=100, patch_size=4, in_chans=12, num_classes=11,
                                                    embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                                    window_size=31, mlp_ratio=4., qkv_bias=True,
                                                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                                    norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                                    use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0])
    """
    """
    net = SwinTransformerV2_1D.SwinTransformerV2_1D(seq_len=100, patch_size=4, in_chans=12, num_classes=11,
                                                    embed_dim=128, depths=[1, 1, 2, 1], num_heads=[1, 2, 4, 8],
                                                    window_size=16, mlp_ratio=2., qkv_bias=True,
                                                    drop_rate=0.5, attn_drop_rate=0.4, drop_path_rate=0.4,
                                                    norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                                    use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0])
    """

    net = SwinTransformerV2_1D.SwinTransformerV2_1D(seq_len=512, patch_size=4, in_chans=12, num_classes=11,
                                                    embed_dim=128, depths=[1, 1, 2, 1], num_heads=[1, 2, 4, 8],
                                                    window_size=16, mlp_ratio=2., qkv_bias=True,
                                                    drop_rate=0.5, attn_drop_rate=0.4, drop_path_rate=0.4,
                                                    norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                                    use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0])

    return net


def get_swin_transformer_v2_1d_experiment(mode="up_sample"):
    if mode == "watch":
        return SwinTransformerV2_1D.SwinTransformerV2_1D(seq_len=512, patch_size=4, in_chans=6, num_classes=11,
                                                         embed_dim=128, depths=[1, 1, 2, 1], num_heads=[1, 2, 4, 8],
                                                         window_size=16, mlp_ratio=2., qkv_bias=True,
                                                         drop_rate=0.5, attn_drop_rate=0.4, drop_path_rate=0.4,
                                                         norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                                         use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0])
    elif mode == "glasses":
        return SwinTransformerV2_1D.SwinTransformerV2_1D(seq_len=102, patch_size=4, in_chans=6, num_classes=11,
                                                         embed_dim=128, depths=[1, 1, 2, 1], num_heads=[1, 2, 4, 8],
                                                         window_size=16, mlp_ratio=2., qkv_bias=True,
                                                         drop_rate=0.5, attn_drop_rate=0.4, drop_path_rate=0.4,
                                                         norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                                         use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0])
    elif mode == "up_sample":
        return SwinTransformerV2_1D.SwinTransformerV2_1D(seq_len=512, patch_size=4, in_chans=12, num_classes=11,
                                                         embed_dim=128, depths=[1, 1, 2, 1], num_heads=[1, 2, 4, 8],
                                                         window_size=16, mlp_ratio=2., qkv_bias=True,
                                                         drop_rate=0.5, attn_drop_rate=0.4, drop_path_rate=0.4,
                                                         norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                                         use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0])
    elif mode == "down_sample":
        return SwinTransformerV2_1D.SwinTransformerV2_1D(seq_len=100, patch_size=4, in_chans=12, num_classes=11,
                                                         embed_dim=128, depths=[1, 1, 2, 1], num_heads=[1, 2, 4, 8],
                                                         window_size=16, mlp_ratio=2., qkv_bias=True,
                                                         drop_rate=0.5, attn_drop_rate=0.4, drop_path_rate=0.4,
                                                         norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                                         use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0])


def get_swin_transformer():
    model = timm.create_model('swin_tiny_patch4_window7_224', img_size=(512, 256))
    model.head.fc = nn.Linear(in_features=768, out_features=11)

    return model


def get_swin_transformer_with_segmentation_head():
    model = timm.create_model('swin_tiny_patch4_window7_224', img_size=(512, 256))
    model.head = SegmentationHead()

    return model


class SegmentationHead(nn.Module):
    def __init__(self):
        super(SegmentationHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1)
        self.transposed_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2,
                                                   padding=1)
        self.transposed_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.transposed_conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.transposed_conv4 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.transposed_conv5 = nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = self.conv1x1(x)

        x = self.transposed_conv1(x)
        x = self.transposed_conv2(x)
        x = self.transposed_conv3(x)
        x = self.transposed_conv4(x)
        x = self.transposed_conv5(x)

        return x
