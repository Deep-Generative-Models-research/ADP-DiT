import math
import torch
import torch.nn as nn
from einops import repeat
from timm.models.layers import to_2tuple
# from .metadataemb_layers import process_metadata  # Import the function

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding

    Image to Patch Embedding using Conv2d

    A convolution based approach to patchifying a 2D image w/ embedding projection.

    Based on the impl in https://github.com/google-research/vision_transformer

    Hacked together by / Copyright 2020 Ross Wightman

    Remove the _assert function in forward function to be compatible with multi-resolution images.
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            img_size = tuple(img_size)
        else:
            raise ValueError(f"img_size must be int or tuple/list of length 2. Got {img_size}")
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def update_image_size(self, img_size):
        self.img_size = img_size
        self.grid_size = (img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def timestep_embedding(t, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = repeat(t, "b -> b d", d=dim)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


# class TimestepEmbedderWithMetadata(nn.Module):
#     """
#     Embeds scalar timesteps and metadata into vector representations.
#     """
#     def __init__(self, hidden_size, frequency_embedding_size=256, metadata_dim=10, out_size=None):
#         super().__init__()
#         if out_size is None:
#             out_size = hidden_size

#         # Timestep embedding
#         self.mlp_t = nn.Sequential(
#             nn.Linear(frequency_embedding_size, hidden_size, bias=True),
#             nn.SiLU(),
#             nn.Linear(hidden_size, out_size, bias=True),
#         )
#         self.frequency_embedding_size = frequency_embedding_size

#         # Metadata embedding
#         self.mlp_metadata = nn.Sequential(
#             nn.Linear(metadata_dim, hidden_size, bias=True),
#             nn.SiLU(),
#             nn.Linear(hidden_size, out_size, bias=True),
#         )

#         # Projection layer to combine timestep and metadata embeddings
#         self.proj = nn.Sequential(
#             nn.Linear(out_size * 2, out_size),
#             nn.SiLU(),
#             nn.Linear(out_size, out_size, bias=True),
#         )

#     def forward(self, t, metadata_csv_path=None, metadata_fields=None):
#         # Embed timestep
#         t_freq = timestep_embedding(t, self.frequency_embedding_size).type(self.mlp_t[0].weight.dtype)
#         t_emb = self.mlp_t(t_freq)

#         # Process and embed metadata
#         if metadata_csv_path and metadata_fields:
#             metadata = process_metadata(metadata_csv_path, metadata_fields)  # Call external function
#             metadata_emb = self.mlp_metadata(metadata)
#         else:
#             raise ValueError("metadata_csv_path and metadata_fields must be provided")

#         # Combine embeddings
#         combined_emb = torch.cat([t_emb, metadata_emb], dim=-1)
#         combined_emb = self.proj(combined_emb)

#         return combined_emb


# class MetadataEmbedder(nn.Module):
#     """Embeds metadata directly."""
#     def __init__(self, hidden_size, metadata_dim=10, out_size=None):
#         super().__init__()
#         if out_size is None:
#             out_size = hidden_size
#         self.mlp_metadata = nn.Sequential(
#             nn.Linear(metadata_dim, hidden_size, bias=True),
#             nn.SiLU(),
#             nn.Linear(hidden_size, out_size, bias=True),
#         )

#     def forward(self, metadata_csv_path, metadata_fields):
#         # Process metadata CSV
#         metadata = process_metadata(metadata_csv_path, metadata_fields)  # Call external function
#         return self.mlp_metadata(metadata)
