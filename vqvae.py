import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        x_flat = x.view(-1, self.embedding_dim)

        distances = (torch.sum(x_flat ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(x_flat, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embeddings(encoding_indices).view(x.shape)

        loss = F.mse_loss(quantized.detach(), x) + self.commitment_cost * F.mse_loss(quantized, x.detach())

        quantized = x + (quantized - x).detach()
        return quantized, loss

# default parameters, input dimension = 2, hidden dimension = 128, number of embeddings = 512, embedding dimension = 64
class VQVAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_embeddings=512, embedding_dim=64):
        super(VQVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss = self.vq_layer(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, vq_loss
