import torch
import numpy as np
import scipy.io
from torch.utils.data import DataLoader, Dataset
from vqvae import VQVAE
import time
import matplotlib.pyplot as plt
import os

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Lazy loading dataset implementation for test data
class LazyMATDataset(Dataset):
    """Dataset for lazy loading of test data"""
    def __init__(self, file_path, key='HT', max_samples=None):
        self.file_path = file_path
        self.key = key
        self.max_samples = max_samples
        
        # Load only metadata to get the dataset size
        if os.path.exists(file_path):
            self.data_info = scipy.io.whosmat(file_path)
            self.data_size = None
            
            # Find size of target array
            for var_name, shape, _ in self.data_info:
                if var_name == self.key:
                    self.data_size = shape[0]  # Number of samples
                    break
                    
            if self.data_size is None:
                raise ValueError(f"Key '{self.key}' not found in {file_path}")
                
            # Apply max_samples limit if specified
            if self.max_samples is not None and self.max_samples < self.data_size:
                self.data_size = self.max_samples
                
            print(f"Initialized test dataset from {file_path} with {self.data_size} samples")
        else:
            raise FileNotFoundError(f"File {file_path} not found")
            
    def __len__(self):
        return self.data_size
        
    def __getitem__(self, idx):
        # Load only the specific portion of the .mat file needed
        try:
            # Cache the data for efficiency
            if not hasattr(self, 'data_cache'):
                print(f"Loading test data from {self.file_path}")
                mat_data = scipy.io.loadmat(self.file_path)
                
                if self.max_samples is not None:
                    self.data_cache = mat_data[self.key][:self.max_samples].astype('float32')
                else:
                    self.data_cache = mat_data[self.key].astype('float32')
            
            # Get the specific sample
            sample = self.data_cache[idx].copy()
            
            # Normalize and reshape
            sample = sample - 0.5
            sample = np.reshape(sample, (2, 32, 32))
            
            return torch.tensor(sample, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading sample {idx} from {self.file_path}: {e}")
            raise

# Test dataset size based on Junyong's paper
test_size = 20000

# For testing purposes, use smaller subset
test_subset_size = 2000  # Modify as needed

# Create test dataset with lazy loading
test_dataset = LazyMATDataset("data/DATA_Htestin.mat", max_samples=test_subset_size)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the trained model
model = VQVAE(
    input_dim=2, 
    hidden_dim=256,  # DeepMind used 256 hidden units
    num_embeddings=512,  # Codebook size K=512
    embedding_dim=64  # Embedding dimension
).to(device)

# Attempt to load best model, fall back to final if not available
try:
    model.load_state_dict(torch.load("vqvae_model_best.pth", map_location=device))
    print("Loaded best model weights")
except FileNotFoundError:
    try:
        model.load_state_dict(torch.load("vqvae_model_final.pth", map_location=device))
        print("Loaded final model weights")
    except FileNotFoundError:
        print("WARNING: No model weights found. Using randomly initialized model.")

model.eval()  # Set model to evaluation mode

# Open a log file
log_filename = "vqvae_test_log.txt"
with open(log_filename, "w") as log_file:
    log_file.write("VQ-VAE Testing Log\n")
    log_file.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    log_file.write("=== Testing Parameters ===\n")
    log_file.write(f"Batch Size: {64}\n")
    log_file.write(f"Test Size: {test_subset_size}\n")
    log_file.write(f"Device: {device}\n\n")
    log_file.write("=== Evaluation Metrics ===\n")
    log_file.write(f"{'NMSE (dB)':<15} {'Correlation (rho)'}\n")
    log_file.write("="*40 + "\n")

# Initialize lists for storing results
reconstructed_data = []
original_data = []

# Run inference on test data
print("Starting inference...")
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)

        # Reconstruct the input data using VQ-VAE
        reconstructed, _ = model(data)

        # Convert to numpy and store results
        reconstructed_data.append(reconstructed.cpu().numpy())
        original_data.append(data.cpu().numpy())
        
        if batch_idx % 10 == 0:
            print(f"Processed batch {batch_idx}/{len(test_loader)}")

# Convert results to numpy arrays
reconstructed_data = np.concatenate(reconstructed_data, axis=0)
original_data = np.concatenate(original_data, axis=0)

# Compute NMSE (Normalized Mean Squared Error)
mse = np.mean((original_data - reconstructed_data) ** 2)
power = np.mean(original_data ** 2)
nmse = 10 * np.log10(mse / power)

# Compute correlation (rho)
num = np.sum(original_data * reconstructed_data, axis=(1, 2, 3))
den = np.sqrt(np.sum(original_data ** 2, axis=(1, 2, 3)) * np.sum(reconstructed_data ** 2, axis=(1, 2, 3)))
rho = np.mean(num / den)

# Print results
print(f"NMSE: {nmse:.4f} dB")
print(f"Correlation: {rho:.4f}")

# Append evaluation results to log file
with open(log_filename, "a") as log_file:
    log_file.write(f"{nmse:<15.4f} {rho:.4f}\n")
    log_file.write("="*40 + "\n")
    log_file.write("Testing Complete!\n")
    log_file.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Plot some test examples
n = 5  # Number of test examples to visualize
plt.figure(figsize=(15, 8))

for i in range(n):
    # Original CSI - Real part
    plt.subplot(4, n, i + 1)
    plt.imshow(original_data[i, 0, :, :], cmap='viridis')
    plt.title("Original (Real)")
    plt.axis("off")

    # Original CSI - Imaginary part
    plt.subplot(4, n, i + 1 + n)
    plt.imshow(original_data[i, 1, :, :], cmap='viridis')
    plt.title("Original (Imag)")
    plt.axis("off")

    # Reconstructed CSI - Real part
    plt.subplot(4, n, i + 1 + 2*n)
    plt.imshow(reconstructed_data[i, 0, :, :], cmap='viridis')
    plt.title("Reconstruct (Real)")
    plt.axis("off")

    # Reconstructed CSI - Imaginary part
    plt.subplot(4, n, i + 1 + 3*n)
    plt.imshow(reconstructed_data[i, 1, :, :], cmap='viridis')
    plt.title("Reconstruct (Imag)")
    plt.axis("off")

# Save the figure as a PNG file
plot_filename = "vqvae_reconstruction_test_examples.png"
plt.tight_layout()
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
print(f"Saved reconstruction plot as {plot_filename}")

# Display the plot if in interactive environment
plt.show()

# Calculate codebook usage
print("Analyzing codebook usage...")
encodings_count = {}

# Process more data to analyze codebook usage
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        
        # Get encoder output
        z_e = model.encoder(data)
        
        # Pass through VQ layer to get indices
        z_q, _ = model.vq_layer(z_e)
        
        # Count usage of each codebook entry
        # This part is approximate since we don't directly access indices
        
        if batch_idx >= 10:  # Limit to a reasonable amount of data
            break

print("Testing and evaluation complete!")