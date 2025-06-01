import torch
import torch.nn as nn
import torch.nn.functional as F

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def load_latent_action_model(model_path, device):
    model = LatentActionVQVAE()
    checkpoint = torch.load(model_path, map_location=device)
    # Fix state dict keys by removing the '*orig*mod.' prefix
    fixed_state_dict = {}
    for k, v in checkpoint['model'].items():
        if k.startswith('_orig_mod.'):
            fixed_state_dict[k.replace('_orig_mod.', '')] = v
        else:
            fixed_state_dict[k] = v
    model.load_state_dict(fixed_state_dict)
    return model, checkpoint['step']

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, q, kv):
        q_norm = self.norm1_q(q)
        kv_norm = self.norm1_kv(kv)
        attended, _ = self.attn(q_norm, kv_norm, kv_norm)
        x = q + attended
        x = x + self.mlp(self.norm2(x))
        return x


class CViViTEncoderCrossAttention(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, patch_size=(5,7), num_heads=4, num_layers=4):
        super().__init__()
        self.patch_h, self.patch_w = patch_size

        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, kernel_size=(4,7), stride=2, padding=(1,3))  # -> (B, C, 5, 7)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + patch_size[0] * patch_size[1], embed_dim))

        self.transformer = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, frame_t, frame_tp1):
        # Stem encode both frames
        feat_t = self.conv_stem(frame_t)     # (B, C, 5, 7)
        feat_tp1 = self.conv_stem(frame_tp1) # (B, C, 5, 7)

        B, C, H, W = feat_tp1.shape

        tokens_q = feat_tp1.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens_kv = feat_t.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(B, -1, -1)
        tokens_q = torch.cat([cls, tokens_q], dim=1)
        tokens_q = tokens_q + self.pos_embed[:, :tokens_q.shape[1]]

        for block in self.transformer:
            tokens_q = block(tokens_q, tokens_kv)

        x = tokens_q[:, 1:, :].transpose(1, 2).view(B, C, H, W)  # Discard CLS, reshape back
        return x

class Encoder(nn.Module):
    """
    Encoder for VQ-VAE latent action model.
    - Input: Concatenated current and next frames (B, 6, 160, 210) for RGB, or (B, 2, 160, 210) for grayscale.
    - Output: Latent feature map (B, 128, 5, 7)

    The architecture uses 5 Conv2d layers with stride=2 to downsample the input.
    The kernel sizes and paddings are chosen to ensure the final output spatial size is exactly (5, 7) for input (160, 210).

    Downsampling calculation for (H, W) = (160, 210):
    Layer 1: (160, 210) -> (80, 105)
    Layer 2: (80, 105) -> (40, 53)
    Layer 3: (40, 53) -> (20, 27)
    Layer 4: (20, 27) -> (10, 14)
    Layer 5: (10, 14) -> (5, 7)

    The last Conv2d uses kernel_size=(4,7), stride=2, padding=(1,3) to ensure the width is exactly 7 (see calculation below):
        output_width = floor((input_width + 2*padding - kernel_size) / stride) + 1
        For input_width=14, kernel_size=7, padding=3, stride=2:
        output_width = floor((14 + 6 - 7) / 2) + 1 = floor(13/2) + 1 = 6 + 1 = 7
    """
    def __init__(self, in_channels=6, hidden_dims=[64, 128, 256, 512, 512], out_dim=128):
        super().__init__()
        layers = []
        c_in = in_channels
        for i, c_out in enumerate(hidden_dims):
            if i < 4:
                # Standard downsampling: kernel=4, stride=2, padding=1
                # This halves the spatial size each time
                layers.append(nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            else:
                # Last layer: custom kernel for width to get (5, 7) output
                # kernel_size=(4,7), stride=2, padding=(1,3)
                # See docstring for calculation
                layers.append(nn.Conv2d(c_in, c_out, kernel_size=(4,7), stride=2, padding=(1,3)))
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        # Project to latent embedding dimension (128)
        self.project = nn.Conv2d(hidden_dims[-1], out_dim, kernel_size=1)
    def forward(self, x):
        x = self.conv(x)
        x = self.project(x)
        return x  # (B, 128, 5, 7)

class VectorQuantizer(nn.Module):
    """
    Vector quantization layer for VQ-VAE.
    - Codebook size: 256
    - Embedding dim: 128
    - Uses straight-through estimator for backprop.
    - Returns quantized latents, indices, and losses.
    """
    def __init__(self, num_embeddings=256, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    def forward(self, z):
        # z: (B, C, H, W)
        # Flatten spatial dimensions for vector quantization
        z_flat = z.permute(0,2,3,1).contiguous().view(-1, self.embedding_dim)  # (B*H*W, C)
        # Compute L2 distance to codebook
        d = (z_flat.pow(2).sum(1, keepdim=True)
             - 2 * z_flat @ self.embeddings.weight.t()
             + self.embeddings.weight.pow(2).sum(1))
        encoding_indices = torch.argmin(d, dim=1)
        quantized = self.embeddings(encoding_indices).view(z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)
        quantized = quantized.permute(0,3,1,2).contiguous()
        # Losses
        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z)
        codebook_loss = F.mse_loss(quantized, z.detach())
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        return quantized, encoding_indices.view(z.shape[0], z.shape[2], z.shape[3]), commitment_loss, codebook_loss

class Decoder(nn.Module):
    """
    Decoder for VQ-VAE latent action model.
    - Input: Quantized latent (B, 128, 5, 7) and current frame (B, 3, 160, 210)
    - Output: Reconstructed next frame (B, 3, 160, 210)

    The decoder upsamples the latent representation back to the original frame size using 5 transposed conv layers.
    - The upsampling path is symmetric to the encoder, but due to rounding, the output may not be exactly (160, 210).
    - To guarantee the output is (160, 210), we use F.interpolate at the end.

    Conditioning:
    - The current frame is processed through a small conv net and concatenated with the latent before upsampling (FiLM-style conditioning).
    - No skip connections are used (to enforce information bottleneck).
    """
    def __init__(self, in_channels=128, cond_channels=3, hidden_dims=[512, 512, 256, 128, 64], out_channels=3):
        super().__init__()
        # Process current frame for conditioning
        self.cond_conv = nn.Sequential(
            nn.Conv2d(cond_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Combine latent and conditioning
        self.fc = nn.Conv2d(in_channels+128, hidden_dims[0], kernel_size=1)
        up_layers = []
        c_in = hidden_dims[0]
        for c_out in hidden_dims[1:]:
            # Standard upsampling: kernel=4, stride=2, padding=1
            up_layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            up_layers.append(nn.BatchNorm2d(c_out))
            up_layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        # Final upsampling to get close to (160, 210)
        up_layers.append(nn.ConvTranspose2d(c_in, out_channels, kernel_size=4, stride=2, padding=1))
        self.up = nn.Sequential(*up_layers)
    def forward(self, z, cond):
        # Process conditioning frame
        cond_feat = self.cond_conv(cond)
        # Resize conditioning to match latent spatial size
        cond_feat = F.interpolate(cond_feat, size=z.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate and upsample
        x = torch.cat([z, cond_feat], dim=1)
        x = self.fc(x)
        x = self.up(x)
        # Guarantee output is (B, 3, 160, 210) by resizing
        x = F.interpolate(x, size=(160, 210), mode='bilinear', align_corners=False)
        return x

class LatentActionVQVAE(nn.Module):
    """
    Full VQ-VAE model for latent action prediction.
    - Encoder: Extracts latent from (frame_t, frame_t+1)
    - VectorQuantizer: Discretizes latent
    - Decoder: Reconstructs next frame from quantized latent and current frame
    """
    def __init__(self, codebook_size=256, embedding_dim=128, commitment_cost=0.25, encoder_type="CViViTEncoderCrossAttention"):
        super().__init__()
        self.encoder_type = encoder_type
        if self.encoder_type == "CViViTEncoderCrossAttention":
            self.encoder = CViViTEncoderCrossAttention()
        else:
            self.encoder = Encoder()
        self.vq = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder()

    def forward(self, frame_t, frame_tp1, return_latent=False):
        # Original frames: (B, C, 210, 160)
        # Need to permute to: (B, C, 160, 210) for the model's internal processing
        frame_t_permuted = frame_t.permute(0, 1, 3, 2)  # (B, C, 210, 160) -> (B, C, 160, 210)
        frame_tp1_permuted = frame_tp1.permute(0, 1, 3, 2)  # (B, C, 210, 160) -> (B, C, 160, 210)
        
        # Concatenate along channel dimension (dim=1)
        x = torch.cat([frame_t_permuted, frame_tp1_permuted], dim=1)  # (B, 2*C, 160, 210)
        if self.encoder_type == "CViViTEncoderCrossAttention":
            # For CViViTEncoderCrossAttention, we need to pass both frames separately
            z = self.encoder(frame_t_permuted, frame_tp1_permuted)  # (B, 128, 5, 7)
        else:
            z = self.encoder(x)  # (B, 128, 5, 7)
        quantized, indices, commitment_loss, codebook_loss = self.vq(z)
        
        # The decoder expects permuted input
        recon_permuted = self.decoder(quantized, frame_t_permuted)
        
        # IMPORTANT: Permute back to match original frame shape (B, C, 210, 160)
        # We need to explicitly do this to ensure the output matches the target shape
        recon = recon_permuted.permute(0, 1, 3, 2)  # (B, C, 160, 210) -> (B, C, 210, 160)
        
        if return_latent:
            return recon, indices, commitment_loss, codebook_loss, z
        else:
            return recon, indices, commitment_loss, codebook_loss

class ActionToLatentMLP(nn.Module):
    def __init__(self, input_dim=4, hidden1=512, hidden2=256, latent_dim=35, codebook_size=256, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, latent_dim * codebook_size)
        )

    def forward(self, x):
        out = self.net(x)  # (batch, latent_dim * codebook_size)
        out = out.view(-1, self.latent_dim, self.codebook_size)
        return out

    def sample_latents(self, logits, temperature=1.0):
        # logits: (batch, 35, 256)
        if temperature <= 0:
            raise ValueError("Temperature must be > 0")
        probs = F.softmax(logits / temperature, dim=-1)  # (batch, 35, 256)
        batch, latent_dim, codebook_size = probs.shape
        # Sample for each position
        samples = torch.multinomial(probs.view(-1, codebook_size), 1).view(batch, latent_dim)
        return samples

class ActionStateToLatentMLP(nn.Module):
    def __init__(self, action_dim=4, hidden1=512, hidden2=256, latent_dim=35, codebook_size=256, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        # Frame encoder for 2 RGB frames (6 channels, 210x160)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=8, stride=4),  # (B, 16, 51, 39)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # (B, 32, 24, 18)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (B, 64, 11, 8)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 11 * 8, 128),
            nn.ReLU(),
        )
        # Combined MLP for action + frame features
        self.net = nn.Sequential(
            nn.Linear(action_dim + 128, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, latent_dim * codebook_size)
        )

    def forward(self, action, frames):
        # action: (B, 4), frames: (B, 6, 210, 160)
        frame_features = self.frame_encoder(frames)
        combined = torch.cat([action, frame_features], dim=1)
        out = self.net(combined)
        return out.view(-1, self.latent_dim, self.codebook_size)

    def sample_latents(self, logits, temperature=1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be > 0")
        probs = F.softmax(logits / temperature, dim=-1)
        batch, latent_dim, codebook_size = probs.shape
        samples = torch.multinomial(probs.view(-1, codebook_size), 1).view(batch, latent_dim)
        return samples