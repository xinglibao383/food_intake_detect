import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)
        return x


class CrossViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(CrossViTBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_output = self.attention(x)
        x = self.norm1(x + attention_output)
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        x = self.dropout(x)
        return x


class CrossViT(nn.Module):
    def __init__(self, in_channels_large, in_channels_small, embed_dim, num_heads, num_classes, num_layers, mlp_dim, dropout=0.1):
        super(CrossViT, self).__init__()
        # 图像patch嵌入层
        self.patch_embed_large = PatchEmbedding(in_channels_large, embed_dim)
        self.patch_embed_small = PatchEmbedding(in_channels_small, embed_dim)
        # 大尺度和小尺度的分类标记
        self.cls_token_large = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_small = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # CrossViT模型的Transformer块
        self.transformer_blocks = nn.Sequential(
            *[CrossViTBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        # 自适应平均池化层
        self.pool = nn.AdaptiveAvgPool1d(1)
        # MLP分类头
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim * 2, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        x_large, x_small = x[0], x[1]

        # 提取大尺度和小尺度的图像特征
        x_large = self.patch_embed_large(x_large)
        x_small = self.patch_embed_small(x_small)

        # 添加分类标记
        cls_token_large = self.cls_token_large.expand(x_large.shape[0], -1, -1)
        cls_token_small = self.cls_token_small.expand(x_small.shape[0], -1, -1)
        x_large = torch.cat((cls_token_large, x_large), dim=1)
        x_small = torch.cat((cls_token_small, x_small), dim=1)

        # 将大尺度和小尺度特征分别输入到各自的Transformer块
        x_large = self.transformer_blocks(x_large)
        x_small = self.transformer_blocks(x_small)

        # 提取分类头的特征，并通过自适应平均池化层进行融合
        x_large = self.pool(x_large.transpose(1, 2)).squeeze(-1)
        x_small = self.pool(x_small.transpose(1, 2)).squeeze(-1)

        # 将大尺度和小尺度的特征堆叠在一起
        x = torch.cat((x_large, x_small), dim=1)

        # 使用MLP进行分类
        x = self.mlp_head(x)
        return x