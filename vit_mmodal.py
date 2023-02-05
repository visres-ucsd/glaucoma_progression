import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# adapted from https://github.com/lucidrains/vit-pytorch

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-06)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 12, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT_rnfl(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, \
    pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., rnfl_len, num_segments):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # (b, 3, 224, 224) -> (b, 3, (14 * 16), (14 * 16)) -> (b, (14 * 14), (16 * 16 * 3))
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        segment_len = int(rnfl_len / num_segments)
        # custom embedding for the 768 RNFL vector 
        # example below num_segments = 48, segment_len = 16
        # (b, 768) -> (b, (48 * 16)) -> (b, 48, 16) -> (b, 48, dim)
        self.to_segment_embedding = nn.Sequential(
            Rearrange('b (n l) -> b n l', n = num_segments),
            nn.Linear(segment_len, dim),
        )

        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1 + num_segments, dim).normal_(std=0.02))  # from BERT
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-06),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img_rnfl):
        img, rnfl =  img_rnfl
        x = self.to_patch_embedding(img)

        x_rnfl = self.to_segment_embedding(rnfl)

        x = torch.cat((x, x_rnfl), dim=1)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class ViT_rnfl_vf(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, \
    pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., rnfl_len=768., rnfl_num_segments, \
    vf_len, vf_num_segments):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # (b, 3, 224, 224) -> (b, 3, (14 * 16), (14 * 16)) -> (b, (14 * 14), (16 * 16 * 3))
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        rnfl_segment_len = int(rnfl_len / rnfl_num_segments)
        # custom embedding for the 768 RNFL vector 
        # example below num_segments = 48, segment_len = 16
        # (b, 768) -> (b, (48 * 16)) -> (b, 48, 16) -> (b, 48, dim)
        self.to_rnfl_segment_embedding = nn.Sequential(
            Rearrange('b (n l) -> b n l', n = rnfl_num_segments),
            nn.Linear(rnfl_segment_len, dim),
        )

        vf_segment_len = int(vf_len / vf_num_segments)
        # custom embedding for the 156 RNFL vector 
        # example below num_segments = 39, segment_len = 4
        # (b, 156) -> (b, (39 * 4)) -> (b, 39, 4) -> (b, 39, dim)
        self.to_vf_segment_embedding = nn.Sequential(
            Rearrange('b (n l) -> b n l', n = vf_num_segments),
            nn.Linear(vf_segment_len, dim),
        )

        # self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1 + num_segments, dim).normal_(std=0.02))  # from BERT
        self.pos_embedding_img = nn.Parameter(torch.empty(1, num_patches + 1, dim).normal_(std=0.02))  # from BERT
        self.pos_embedding_rnfl = nn.Parameter(torch.empty(1, rnfl_num_segments, dim).normal_(std=0.02))  # from BERT
        self.pos_embedding_vf = nn.Parameter(torch.empty(1, vf_num_segments, dim).normal_(std=0.02))  # from BERT

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-06),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img_rnfl_vf):
        img, rnfl, vf =  img_rnfl_vf
        
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding_img[:, :(n + 1)]

        x_rnfl = self.to_rnfl_segment_embedding(rnfl)
        _, n, _ = x_rnfl.shape
        x_rnfl += self.pos_embedding_rnfl[:, :(n + 1)]

        x_vf = self.to_vf_segment_embedding(vf)
        _, n, _ = x_vf.shape
        x_vf += self.pos_embedding_vf[:, :(n + 1)]

        x = torch.cat((x, x_rnfl, x_vf), dim=1)
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class ViT_rnfl_vf_longitudinal(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, \
    pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., rnfl_len=768., rnfl_num_segments, \
    vf_len, vf_num_segments):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # (b, 3, 224, 224) -> (b, 3, (14 * 16), (14 * 16)) -> (b, (14 * 14), (16 * 16 * 3))
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        rnfl_segment_len = int(rnfl_len / rnfl_num_segments)
        # custom embedding for the 768 RNFL vector 
        # example below num_segments = 48, segment_len = 16
        # (b, 768) -> (b, (48 * 16)) -> (b, 48, 16) -> (b, 48, dim)
        self.to_rnfl_segment_embedding = nn.Sequential(
            Rearrange('b (n l) -> b n l', n = rnfl_num_segments),
            nn.Linear(rnfl_segment_len, dim),
        )

        vf_segment_len = int(vf_len / vf_num_segments)
        # custom embedding for the 156 RNFL vector 
        # example below num_segments = 39, segment_len = 4
        # (b, 156) -> (b, (39 * 4)) -> (b, 39, 4) -> (b, 39, dim)
        self.to_vf_segment_embedding = nn.Sequential(
            Rearrange('b (n l) -> b n l', n = vf_num_segments),
            nn.Linear(vf_segment_len, dim),
        )

        # self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1 + num_segments, dim).normal_(std=0.02))  # from BERT
        self.pos_embedding_img = nn.Parameter(torch.empty(1, num_patches * 2 + 1, dim).normal_(std=0.02))  # from BERT
        self.pos_embedding_rnfl = nn.Parameter(torch.empty(1, rnfl_num_segments * 2, dim).normal_(std=0.02))  # from BERT
        self.pos_embedding_vf = nn.Parameter(torch.empty(1, vf_num_segments * 2, dim).normal_(std=0.02))  # from BERT

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-06),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img_rnfl_vf):
        img, img2, rnfl, rnfl2, vf, vf2, time_delta =  img_rnfl_vf
        
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x2 = self.to_patch_embedding(img2)
        x = torch.cat((cls_tokens, x, x2), dim=1)
        x += self.pos_embedding_img[:, :(n * 2 + 1)]

        x_rnfl = self.to_rnfl_segment_embedding(rnfl)
        x_rnfl2 = self.to_rnfl_segment_embedding(rnfl2)
        x_rnfl = torch.cat((x_rnfl, x_rnfl2), dim=1)
        _, n, _ = x_rnfl.shape
        x_rnfl += self.pos_embedding_rnfl[:, :(n + 1)]

        x_vf = self.to_vf_segment_embedding(vf)
        x_vf2 = self.to_vf_segment_embedding(vf2)
        x_vf = torch.cat((x_vf, x_vf2), dim=1)
        _, n, _ = x_vf.shape
        x_vf += self.pos_embedding_vf[:, :(n + 1)]

        x_tdelta = torch.repeat_interleave(time_delta, x_vf.shape[2], axis=2)

        x = torch.cat((x, x_rnfl, x_vf, x_tdelta), dim=1)
        # x = torch.cat((x_vf, x_tdelta), dim=1)
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)