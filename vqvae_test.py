import torch
import numpy as np
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
from vqvae import VQVAE
import time

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE(input_dim=2).to(device)
model.load_state_dict(torch.load("vqvae_model.pth"))  # Load trained weights
model.eval()  # Set model to evaluation mode

# Function to load test data
def load_data(filepath, num_samples=None):
    mat = scipy.io.loadmat(filepath)
    data = mat['HT'].astype('float32')

    print(f"Loaded test data shape from {filepath}: {data.shape}")  # Debugging

    # If `num_samples` is provided, use only a subset of the data
    if num_samples is not None:
        data = data[:num_samples]

    # Normalize and reshape from (batch, 2048) → (batch, 2, 32, 32)
    data = data - 0.5
    data = np.reshape(data, (data.shape[0], 2, 32, 32))

    print(f"Final reshaped test data shape: {data.shape}")  # Debugging
    return data

# Load test dataset
test_subset_size = 2000  # Modify if needed
test_data = load_data("data/DATA_Htestin.mat", test_subset_size)

# Convert to PyTorch tensors
test_tensor = torch.tensor(test_data, dtype=torch.float32)
test_loader = DataLoader(TensorDataset(test_tensor), batch_size=64)

# Open a log file
log_filename = "vqvae_test_log.txt"
with open(log_filename, "w") as log_file:
    log_file.write("VQ-VAE Testing Log\n")
    log_file.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    log_file.write("=== Testing Parameters ===\n")
    log_file.write(f"Batch Size: {64}\n")
    log_file.write(f"Test Subset Size: {test_subset_size}\n\n")
    log_file.write("=== Evaluation Metrics ===\n")
    log_file.write(f"{'NMSE (dB)':<15} {'Correlation (rho)'}\n")
    log_file.write("="*40 + "\n")

# Initialize lists for storing results
reconstructed_data = []
original_data = []

# Run inference on test data
with torch.no_grad(): 
    for batch in test_loader:
        data = batch[0].to(device)

        # Reconstruct the input data using VQ-VAE
        reconstructed, _ = model(data)

        # Convert to numpy and store results
        reconstructed_data.append(reconstructed.cpu().numpy())
        original_data.append(data.cpu().numpy())

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
import matplotlib.pyplot as plt

n = 5  # Number of test examples to visualize
plt.figure(figsize=(12, 6))

for i in range(n):
    # Original CSI
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(np.abs(original_data[i, 0, :, :]), cmap='gray')
    plt.title("Original CSI")
    plt.axis("off")

    # Reconstructed CSI
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(np.abs(reconstructed_data[i, 0, :, :]), cmap='gray')
    plt.title("Reconstructed CSI")
    plt.axis("off")

# Save the figure as a PNG file
plot_filename = "vqvae_reconstruction_test_examples.png"
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
print(f"Saved reconstruction plot as {plot_filename}")

# Display the plot
plt.show()
