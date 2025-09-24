import time
import math
import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbeddings(nn.Module):

    def __init__(self, input_dim, num_channels, patch_size, model_dim):
        super().__init__()
        assert input_dim % patch_size == 0
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.num_patches = (input_dim // patch_size) ** 2

        patch_dim = num_channels * patch_size * patch_size
        self.in_layer = nn.Linear(patch_dim, model_dim, bias=True)

    def forward(self, x):
        bs, c, h, w = x.shape
        ps = self.patch_size
        x = x.view(bs, c, h // ps, ps, w // ps, ps)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(bs, (h // ps) * (w // ps), c * ps * ps)
        x = self.in_layer(x)
        return x


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, num_positions, model_dim):
        super().__init__()
        position = torch.arange(num_positions).unsqueeze(1)
        freqs = torch.exp(
            torch.arange(0, model_dim, 2) * -(math.log(10000.0) / model_dim)
        )
        pe = torch.zeros(num_positions, model_dim)
        pe[:, 0::2] = torch.sin(position * freqs)
        pe[:, 1::2] = torch.cos(position * freqs)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]


class Embeddings(nn.Module):

    def __init__(self, input_dim, num_channels, patch_size, model_dim):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(
            input_dim, num_channels, patch_size, model_dim
        )
        self.position_encoding = SinusoidalPositionalEncoding(
            self.patch_embeddings.num_patches, model_dim
        )

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = self.position_encoding(x)
        return x


class SelfAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.to_query = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.to_key = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.to_value = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)
        out = F.scaled_dot_product_attention(query, key, value)
        out = self.out_layer(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, dim, ff_dim):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))


class TransformerEncoderBlock(nn.Module):

    def __init__(self, model_dim, ff_dim):
        super().__init__()
        self.self_attention = SelfAttention(model_dim)
        self.self_attention_norm = nn.LayerNorm(model_dim)

        self.feed_forward = FeedForward(model_dim, ff_dim)
        self.feed_forward_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.self_attention_norm(x + self.self_attention(x))
        x = self.feed_forward_norm(x + self.feed_forward(x))
        return x


class VisionTransformer(nn.Module):

    def __init__(
        self,
        input_dim=32,
        num_channels=3,
        patch_size=4,
        model_dim=64,
        ff_dim=256,
        num_blocks=4,
        num_classes=100,
    ):
        super().__init__()
        self.embeddings = Embeddings(input_dim, num_channels, patch_size, model_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(model_dim, ff_dim) for _ in range(num_blocks)]
        )
        self.out_layer = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = self.embeddings(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        logits = self.out_layer(x.mean(dim=1))
        return logits


def get_vit(conf):
    vit = VisionTransformer(**conf)
    return vit


if __name__ == "__main__":
    model = VisionTransformer().to("cuda", torch.bfloat16)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    x = torch.randn(1, 3, 32, 32).to("cuda")
    start_time = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x)
    end_time = time.perf_counter()
    print("Logits shape:", y.shape)
    print(f"Forward time (per one sample): {(end_time - start_time):.4f} s")
