import torch
import torch.optim as optim
import numpy as np
import scipy.io
from torch.utils.data import DataLoader, Dataset
from vqvae import VQVAE
import time
import os
import sys

# Simple command line arguments  
dataset = 'indoor'  # Change to 'outdoor' for outdoor trials
latent_dim = 1024   # Change this for different M values
codebook_size = 512 # Change this for different K values

# Override with command line if provided
if len(sys.argv) > 1:
    dataset = sys.argv[1]
if len(sys.argv) > 2:
    latent_dim = int(sys.argv[2])
if len(sys.argv) > 3:
    codebook_size = int(sys.argv[3])

print(f"Training: Dataset={dataset}, M={latent_dim}, K={codebook_size}")

# Lazy loading dataset implementation
class LazyMATDataset(Dataset):
    """Dataset for loading MIMO channel data from .mat files only when needed"""
    def __init__(self, file_path, key='HT'):
        self.file_path = file_path
        self.key = key
        
        # Load only metadata to get the dataset size
        self.data_info = scipy.io.whosmat(file_path)
        self.data_size = None
        
        # Find size of target array
        for var_name, shape, _ in self.data_info:
            if var_name == self.key:
                self.data_size = shape[0]  # Number of samples
                break
                
        if self.data_size is None:
            raise ValueError(f"Key '{self.key}' not found in {file_path}")
        
        print(f"Initialized dataset from {file_path} with {self.data_size} samples")
            
    def __len__(self):
        return self.data_size
        
    def __getitem__(self, idx):
        if not hasattr(self, 'data_cache'):
            print(f"Loading data from {self.file_path}")
            mat_data = scipy.io.loadmat(self.file_path)
            self.data_cache = mat_data[self.key].astype('float32')
        
        # Get the specific sample
        sample = self.data_cache[idx].copy()
        
        # Normalize and reshape as before
        sample = sample - 0.5
        sample = np.reshape(sample, (2, 32, 32))
        
        return torch.tensor(sample, dtype=torch.float32)

# Set dataset sizes according to Junyong's paper
train_size = 100000
val_size = 30000

# For testing purposes (set to False for full dataset)
use_full_dataset = True
if not use_full_dataset:
    train_size = 5000
    val_size = 1000

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create datasets with lazy loading based on dataset argument
if dataset == 'indoor':
    train_dataset = LazyMATDataset("data/DATA_Htrainin.mat")
    val_dataset = LazyMATDataset("data/DATA_Hvalin.mat")
else:
    train_dataset = LazyMATDataset("data/DATA_Htrainout.mat")
    val_dataset = LazyMATDataset("data/DATA_Hvalout.mat")

# If we need to use a subset of the data
if len(train_dataset) > train_size:
    train_indices = torch.randperm(len(train_dataset))[:train_size]
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

if len(val_dataset) > val_size:
    val_indices = torch.randperm(len(val_dataset))[:val_size]
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

# Create DataLoaders with DeepMind batch size
batch_size = 128  # DeepMind paper (changed from 256)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Set up model with variable parameters
model = VQVAE(input_dim=2,
              hidden_dim=256,  # DeepMind used 256 hidden units
              num_embeddings=codebook_size,  # Now variable
              embedding_dim=latent_dim).to(device)  # Now variable

# Optimizer with Adam as in DeepMind paper
learning_rate = 2e-4  # Learning rate from DeepMind paper
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# Create model directory for this configuration
model_dir = f"models_{dataset}_M{latent_dim}_K{codebook_size}"
os.makedirs(model_dir, exist_ok=True)

# Open a log file
log_filename = f"{model_dir}/training_log.txt"
with open(log_filename, "w") as log_file:
    log_file.write("VQ-VAE Training Log\n")
    log_file.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    log_file.write("=== Training Parameters ===\n")
    log_file.write(f"Dataset: {dataset}\n")
    log_file.write(f"Latent Dimension (M): {latent_dim}\n")
    log_file.write(f"Codebook Size (K): {codebook_size}\n")
    log_file.write(f"Learning Rate: {learning_rate}\n")
    log_file.write(f"Batch Size: {batch_size}\n")
    log_file.write(f"Train Size: {train_size}\n")
    log_file.write(f"Validation Size: {val_size}\n")
    log_file.write("=== Training Progress ===\n")
    log_file.write(f"{'Epoch':<10} {'Recon Loss':<15} {'VQ Loss':<15} {'Val Loss'}\n")
    log_file.write("="*50 + "\n")

# Early stopping logic
patience = 5
min_delta = 1e-5
epochs_no_improve = 0

# Training loop
num_epochs = 50 if use_full_dataset else 10
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss, total_vq_loss = 0, 0
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        
        optimizer.zero_grad()
        reconstructed, vq_loss = model(data)
        
        # Compute loss
        recon_loss = criterion(reconstructed, data)
        loss = recon_loss + vq_loss
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        total_loss += recon_loss.item() 
        total_vq_loss += vq_loss.item()
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, "
                  f"Loss: {recon_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            reconstructed, _ = model(data)
            val_loss += criterion(reconstructed, data).item()
    
    # Print CUDA memory usage after each epoch
    if torch.cuda.is_available():
        print(f"CUDA Memory Usage: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    
    # Average losses
    avg_train_loss = total_loss / len(train_loader)
    avg_vq_loss = total_vq_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Recon Loss: {avg_train_loss:.4f}, "
          f"VQ Loss: {avg_vq_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}")
    
    # Early stopping logic
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), f"{model_dir}/model_best.pth")
        print("  Saved best model!")
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break
    
    # Append results to log file
    with open(log_filename, "a") as log_file:
        log_file.write(f"{epoch+1:<10} {avg_train_loss:<15.4f} {avg_vq_loss:<15.4f} {avg_val_loss:.4f}\n")
        
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, f"{model_dir}/checkpoint_epoch_{epoch+1}.pth")

# Save the final model
if epochs_no_improve < patience:
    torch.save(model.state_dict(), f"{model_dir}/model_final.pth")

# Append final message to log file
with open(log_filename, "a") as log_file:
    log_file.write("="*50 + "\n")
    log_file.write("Training Complete!\n")
    log_file.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
    log_file.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"Training complete! Models saved in {model_dir}/")
print(f"Best validation loss: {best_val_loss:.4f}")