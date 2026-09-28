"""Checkpoint-compatible DenseTRF image model (no auxiliary model allocation)."""
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from densetrf._vendor.dinov3.models.vision_transformer import DinoVisionTransformer
from .slot_attention import SlotAttention


class DinoFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.dinov3 = DinoVisionTransformer(
            img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
            num_heads=12, ffn_ratio=4, qkv_bias=True, drop_path_rate=0.0,
            layerscale_init=1e-5, norm_layer="layernormbf16", ffn_layer="mlp",
            ffn_bias=True, proj_bias=True, n_storage_tokens=4, mask_k_bias=True,
            pos_embed_rope_base=100, pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32",
        )
        # These aliases/unused projection parameters are present in the published checkpoint.
        # image_encoder references the SAME backbone; it allocates no second backbone.
        self.image_encoder = self.dinov3
        self.output_transform = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 64))

    def forward(self, images):
        return self.image_encoder.forward_features(images)["x_norm_patchtokens"]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([DinoFeatures()])

    def forward(self, images):
        return self.encoders[0](images)


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(256)  # retained for strict checkpoint compatibility
        self.fusion_layer = nn.Sequential(nn.Linear(768, 1536), nn.ReLU(), nn.Linear(1536, 256))

    def forward(self, features):
        return self.fusion_layer(self.norm(features))


class RandomInit(nn.Module):
    """VideoSAUR random slot initialization; see licenses/VideoSAUR.txt."""
    def __init__(self, n_slots=9, dim=256):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.mean = nn.Parameter(torch.zeros(1, 1, dim))
        self.log_std = nn.Parameter(torch.log(torch.ones(1, 1, dim) * dim ** -0.5))

    def forward(self, batch_size):
        noise = torch.randn(batch_size, self.n_slots, self.dim, device=self.mean.device)
        return self.mean + noise * self.log_std.exp()


class Processor(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Module()
        self.module.corrector = SlotAttention(256, 256, n_iters=3, use_mlp=False)

    def forward(self, slots, features):
        return self.module.corrector(slots, features)["slots"]


class Decoder(nn.Module):
    """VideoSAUR-style spatial slot decoder with the original state-dict names."""
    def __init__(self):
        super().__init__()
        self.module = nn.Module()
        self.module.pos_emb = nn.Parameter(torch.randn(1, 1, 196, 256) * 256 ** -0.5)
        self.module.mlp = nn.Module()
        dims = [256, 1024, 1024, 1024, 1024, 769]
        layers = []
        for index, (left, right) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
        self.module.mlp.layers = nn.Sequential(*layers)

    def forward(self, slots):
        predictions = self.module.mlp.layers(slots.unsqueeze(2) + self.module.pos_emb)
        features, alpha = predictions.split((768, 1), dim=-1)
        masks = alpha.softmax(dim=1)
        return (features * masks).sum(dim=1), masks.squeeze(-1)


class DenseTRF(nn.Module):
    components = ("encoder", "initializer", "adapter", "processor", "decoder")

    def __init__(self, num_classes=None):
        super().__init__()
        self.encoder = Encoder()
        self.initializer = RandomInit()
        self.adapter = Adapter()
        self.processor = Processor()
        self.decoder = Decoder()
        self.num_classes = num_classes
        self.dense_head = None
        if num_classes is not None:
            self.dense_head = nn.Module()
            self.dense_head.net = nn.Sequential(nn.Linear(1033, 512), nn.ReLU(), nn.Dropout(0), nn.Linear(512, num_classes))
            for layer in self.dense_head.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def from_features(self, features, *, head_detach=False, initial_slots=None):
        adapted = self.adapter(features)
        initial = self.initializer(features.shape[0]) if initial_slots is None else initial_slots
        slots = self.processor(initial, adapted)
        reconstruction, masks = self.decoder(slots)
        output = dict(features=adapted, reconstruction=reconstruction, masks=masks, slots=slots)
        output["reconstruction_loss"] = F.mse_loss(reconstruction, features.detach())
        if self.dense_head is not None:
            combined = torch.cat([adapted, reconstruction, masks.transpose(1, 2)], dim=-1)
            if head_detach:
                combined = combined.detach()
            logits = self.dense_head.net(combined).transpose(1, 2).reshape(-1, self.num_classes, 14, 14)
            output["logits"] = logits
        return output

    def forward(self, images, *, encoder_grad=False, head_detach=False):
        """images: float Bx3xHxW, values 0..255 in the configured channel order."""
        size = images.shape[-2:]
        resized = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        normalized = (resized - 124.0) / 60.0
        with nullcontext() if encoder_grad else torch.no_grad():
            features = self.encoder(normalized)
        output = self.from_features(features, head_detach=head_detach)
        if "logits" in output:
            output["logits"] = F.interpolate(output["logits"], size=size, mode="bilinear", align_corners=False)
        return output

    def shared_state(self):
        # Preserve aliases in the legacy encoder checkpoint instead of doubling its size.
        tensors = {}
        state = {}
        for name in self.components:
            state[name] = {}
            for key, value in getattr(self, name).state_dict().items():
                identity = (value.device, value.data_ptr(), value.shape, value.stride(), value.dtype)
                if identity not in tensors:
                    tensors[identity] = value.detach().cpu().clone()
                state[name][key] = tensors[identity]
        return state

    def load_shared(self, state):
        if set(state) != set(self.components):
            raise ValueError(f"Expected exactly these shared components: {self.components}")
        for name in self.components:
            getattr(self, name).load_state_dict(state[name], strict=True)
