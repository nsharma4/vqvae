import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual block as described in DeepMind's VQ-VAE paper Section 4.1:
    "implemented as ReLU, 3x3 conv, ReLU, 1x1 conv"
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return x + self.block(x)

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module for VQ-VAE with commitment cost β = 0.25
    Following DeepMind's implementation
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # Loss (DeepMind Equation 3)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss

class VQVAE(nn.Module):
    """
    VQ-VAE model following DeepMind's architecture from Section 4.1
    """
    def __init__(self, input_dim=2, hidden_dim=256, num_embeddings=512, embedding_dim=64):
        super(VQVAE, self).__init__()
        
        # Encoder: 2 strided convolutional layers with stride 2 and window size 4×4,
        # followed by two residual 3×3 blocks
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, embedding_dim, kernel_size=1, stride=1, padding=0)
        )
        
        # Vector quantizer
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost=0.25)
        
        # Decoder: two residual 3×3 blocks, followed by two transposed convolutions 
        # with stride 2 and window size 4×4
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim//2, input_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output scaled between -1 and 1
        )

    def encode(self, x):
        z = self.encoder(x)
        return z
        
    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        quantized, vq_loss = self.vq_layer(z)
        x_recon = self.decode(quantized)
        return x_recon, vq_loss