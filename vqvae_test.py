import torch
import numpy as np
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
from vqvae import VQVAE 

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE(input_dim=2).to(device)
model.load_state_dict(torch.load("vqvae_model.pth"))  # Load trained weights
model.eval()  # Set model to evaluation mode

# Function to load test data
def load_data(filepath):
    mat = scipy.io.loadmat(filepath)
    data = mat['HT'].astype('float32')

    print(f"Loaded test data shape: {data.shape}")  # Debugging

    # Normalize and reshape from (batch, 1, 2048) → (batch, 2, 32, 32)
    data = data - 0.5
    data = np.reshape(data, (data.shape[0], 2, 32, 32))

    return data

# Load test dataset
test_data = load_data("data/DATA_Htestin.mat")

# Convert to PyTorch tensors
test_tensor = torch.tensor(test_data, dtype=torch.float32)
test_loader = DataLoader(TensorDataset(test_tensor), batch_size=256)  # Faster evaluation


# Initialize lists for storing results
reconstructed_data = []
original_data = []

# Run inference on test data
with torch.no_grad():  # No gradients needed for testing
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

# Save results as .mat file
scipy.io.savemat("vqvae_reconstructed.mat", {"reconstructed": reconstructed_data})
scipy.io.savemat("vqvae_original.mat", {"original": original_data})

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

plt.show()
