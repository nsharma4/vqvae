import torch
import numpy as np
import scipy.io
from torch.utils.data import DataLoader, Dataset
from vqvae import VQVAE
import time
import matplotlib.pyplot as plt
import os
import sys

# Simple command line arguments
dataset = 'indoor'
latent_dim = 1024
codebook_size = 512

# Override with command line if provided
if len(sys.argv) > 1:
    dataset = sys.argv[1]
if len(sys.argv) > 2:
    latent_dim = int(sys.argv[2])
if len(sys.argv) > 3:
    codebook_size = int(sys.argv[3])

print(f"Testing: Dataset={dataset}, M={latent_dim}, K={codebook_size}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Lazy loading dataset
class LazyMATDataset(Dataset):
    def __init__(self, file_path, key='HT', max_samples=None):
        self.file_path = file_path
        self.key = key
        self.max_samples = max_samples
        
        if os.path.exists(file_path):
            self.data_info = scipy.io.whosmat(file_path)
            self.data_size = None
            
            for var_name, shape, _ in self.data_info:
                if var_name == self.key:
                    self.data_size = shape[0]
                    break
                    
            if self.data_size is None:
                raise ValueError(f"Key '{self.key}' not found in {file_path}")
                
            if self.max_samples is not None and self.max_samples < self.data_size:
                self.data_size = self.max_samples
                
            print(f"Initialized test dataset with {self.data_size} samples")
        else:
            raise FileNotFoundError(f"File {file_path} not found")
            
    def __len__(self):
        return self.data_size
        
    def __getitem__(self, idx):
        if not hasattr(self, 'data_cache'):
            print(f"Loading test data from {self.file_path}")
            mat_data = scipy.io.loadmat(self.file_path)
            
            if self.max_samples is not None:
                self.data_cache = mat_data[self.key][:self.max_samples].astype('float32')
            else:
                self.data_cache = mat_data[self.key].astype('float32')
        
        sample = self.data_cache[idx].copy()
        sample = sample - 0.5
        sample = np.reshape(sample, (2, 32, 32))
        
        return torch.tensor(sample, dtype=torch.float32)

# Test file and size
if dataset == 'indoor':
    test_file = "data/DATA_Htestin.mat"
else:
    test_file = "data/DATA_Htestout.mat"

test_size = 20000  # Full test size from Junyong's paper

# Model directory
model_dir = f"models_{dataset}_M{latent_dim}_K{codebook_size}"

# Create test dataset
test_dataset = LazyMATDataset(test_file, max_samples=test_size)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load model
model = VQVAE(
    input_dim=2, 
    hidden_dim=256,
    num_embeddings=codebook_size,
    embedding_dim=latent_dim
).to(device)

# Load model weights
best_model_path = f"{model_dir}/model_best.pth"
final_model_path = f"{model_dir}/model_final.pth"

if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("Loaded best model weights")
elif os.path.exists(final_model_path):
    model.load_state_dict(torch.load(final_model_path, map_location=device))
    print("Loaded final model weights")
else:
    print("WARNING: No model weights found. Using randomly initialized model.")

model.eval()

# Run inference
print("Starting inference...")
reconstructed_data = []
original_data = []

with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        reconstructed, _ = model(data)
        
        reconstructed_data.append(reconstructed.cpu().numpy())
        original_data.append(data.cpu().numpy())
        
        if batch_idx % 10 == 0:
            print(f"Processed batch {batch_idx}/{len(test_loader)}")

# Calculate metrics
reconstructed_data = np.concatenate(reconstructed_data, axis=0)
original_data = np.concatenate(original_data, axis=0)

mse = np.mean((original_data - reconstructed_data) ** 2)
power = np.mean(original_data ** 2)
nmse = 10 * np.log10(mse / power)

num = np.sum(original_data * reconstructed_data, axis=(1, 2, 3))
den = np.sqrt(np.sum(original_data ** 2, axis=(1, 2, 3)) * np.sum(reconstructed_data ** 2, axis=(1, 2, 3)))
rho = np.mean(num / den)

print(f"\n{'='*50}")
print(f"RESULTS")
print(f"{'='*50}")
print(f"Dataset: {dataset}")
print(f"Latent Dimension (M): {latent_dim}")
print(f"Codebook Size (K): {codebook_size}")
print(f"NMSE: {nmse:.4f} dB")
print(f"Correlation: {rho:.4f}")
print(f"{'='*50}")

# Save results
with open(f"{model_dir}/test_results.txt", "w") as f:
    f.write(f"NMSE: {nmse:.4f} dB\n")
    f.write(f"Correlation: {rho:.4f}\n")

print(f"Results saved to {model_dir}/test_results.txt")

print("Testing complete!")