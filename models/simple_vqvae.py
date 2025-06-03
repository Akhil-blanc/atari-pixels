import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEncoder(nn.Module):
    """Simple encoder for both frame and action inputs"""
    def __init__(self, input_channels=3, action_dim=4, latent_dim=128):
        super().__init__()
        
        # Frame encoder
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),  # 105x80
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 52x40
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 26x20
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, 4, 2, 1),  # 13x10
            nn.ReLU()
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        
        # Fusion layer
        self.fusion = nn.Conv2d(latent_dim * 2, latent_dim, 1)
        
    def forward(self, frame, action):
        # Encode frame
        frame_feat = self.frame_encoder(frame)  # (B, 128, 13, 10)
        
        # Encode action and broadcast to spatial dimensions
        action_feat = self.action_encoder(action)  # (B, 128)
        action_feat = action_feat.unsqueeze(-1).unsqueeze(-1)  # (B, 128, 1, 1)
        action_feat = action_feat.expand(-1, -1, frame_feat.size(2), frame_feat.size(3))  # (B, 128, 13, 10)
        
        # Combine features
        combined = torch.cat([frame_feat, action_feat], dim=1)  # (B, 256, 13, 10)
        z = self.fusion(combined)  # (B, 128, 13, 10)
        
        return z

class SimpleVectorQuantizer(nn.Module):
    """Simplified vector quantizer with EMA updates"""
    def __init__(self, num_embeddings=256, embedding_dim=128, commitment_cost=0.25, ema_decay=0.99):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        
        # Codebook
        self.register_buffer('codebook', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('ema_count', torch.zeros(num_embeddings))
        self.register_buffer('ema_weight', self.codebook.clone())
        
        # Initialize codebook
        self.codebook.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, z):
        # Flatten spatial dimensions
        z_flat = z.permute(0,2,3,1).contiguous().view(-1, self.embedding_dim)
        
        # Compute distances to codebook
        distances = torch.sum(z_flat**2, dim=1, keepdim=True) + \
                   torch.sum(self.codebook**2, dim=1) - \
                   2 * torch.matmul(z_flat, self.codebook.t())
        
        # Find closest codebook entries
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.num_embeddings).float()
        
        # Quantize
        quantized_flat = torch.matmul(encodings, self.codebook)
        quantized = quantized_flat.view(z.shape)
        
        # EMA updates during training
        if self.training:
            with torch.no_grad():
                # Update EMA statistics
                encodings_sum = encodings.sum(0)
                dw = torch.matmul(encodings.t(), z_flat)
                
                self.ema_count.mul_(self.ema_decay).add_(encodings_sum, alpha=1-self.ema_decay)
                self.ema_weight.mul_(self.ema_decay).add_(dw, alpha=1-self.ema_decay)
                
                # Update codebook
                n = self.ema_count.sum()
                weights = (self.ema_count + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                self.codebook.copy_(self.ema_weight / weights.unsqueeze(1))
        
        # Compute losses
        commitment_loss = F.mse_loss(quantized.detach(), z)
        codebook_loss = F.mse_loss(quantized, z.detach())
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        return quantized, indices.view(z.shape[0], z.shape[2], z.shape[3]), commitment_loss, codebook_loss

class SimpleDecoder(nn.Module):
    """Simple decoder/dynamics model"""
    def __init__(self, latent_dim=128, output_channels=3):
        super().__init__()
        
        self.decoder = nn.Sequential(
            # Upsample from 13x10 to 26x20
            nn.ConvTranspose2d(latent_dim, 256, 4, 2, 1),
            nn.ReLU(),
            # Upsample from 26x20 to 52x40
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            # Upsample from 52x40 to 105x80
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            # Upsample from 105x80 to 210x160
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1),
        )
        
    def forward(self, quantized_z):
        return self.decoder(quantized_z)

class SimplifiedVQVAE(nn.Module):
    """
    Simplified VQ-VAE matching the architecture diagram:
    last_frame + action -> Encode -> z -> Codebook -> Decoder -> Predicted Next Frame
    """
    def __init__(self, 
                 input_channels=3, 
                 action_dim=4, 
                 codebook_size=256, 
                 latent_dim=128,
                 commitment_cost=0.25,
                 ema_decay=0.99):
        super().__init__()
        
        self.encoder = SimpleEncoder(input_channels, action_dim, latent_dim)
        self.vq = SimpleVectorQuantizer(codebook_size, latent_dim, commitment_cost, ema_decay)
        self.decoder = SimpleDecoder(latent_dim, input_channels)
        
    def forward(self, last_frame, action):
        """
        Args:
            last_frame: (B, C, H, W) - current frame
            action: (B, action_dim) - action taken
        
        Returns:
            predicted_next_frame: (B, C, H, W) - predicted next frame
            indices: (B, H', W') - codebook indices
            commitment_loss: scalar - commitment loss
            codebook_loss: scalar - codebook loss
        """
        # Encode inputs
        z = self.encoder(last_frame, action)
        
        # Vector quantization
        quantized_z, indices, commitment_loss, codebook_loss = self.vq(z)
        
        # Decode to next frame
        predicted_next_frame = self.decoder(quantized_z)
        
        return predicted_next_frame, indices, commitment_loss, codebook_loss
    
    def encode_to_indices(self, last_frame, action):
        """Encode inputs to discrete indices"""
        z = self.encoder(last_frame, action)
        _, indices, _, _ = self.vq(z)
        return indices
    
    def decode_from_indices(self, indices):
        """Decode from discrete indices to frame"""
        # Get quantized vectors from indices
        quantized_flat = F.embedding(indices.flatten(), self.vq.codebook)
        quantized = quantized_flat.view(indices.size(0), indices.size(1), indices.size(2), -1)
        quantized = quantized.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Decode to frame
        return self.decoder(quantized)

# Example usage
if __name__ == "__main__":
    # Create model
    model = SimplifiedVQVAE(
        input_channels=3,
        action_dim=4,
        codebook_size=256,
        latent_dim=128
    )
    
    # Example inputs
    batch_size = 4
    last_frame = torch.randn(batch_size, 3, 210, 160)  # Current frame
    action = torch.randn(batch_size, 4)  # Action vector
    
    # Forward pass
    predicted_frame, indices, commitment_loss, codebook_loss = model(last_frame, action)
    
    print(f"Input frame shape: {last_frame.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Predicted frame shape: {predicted_frame.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Commitment loss: {commitment_loss.item():.4f}")
    print(f"Codebook loss: {codebook_loss.item():.4f}")
    
    # Test encoding/decoding
    indices_only = model.encode_to_indices(last_frame, action)
    reconstructed = model.decode_from_indices(indices_only)
    print(f"Reconstructed shape: {reconstructed.shape}")