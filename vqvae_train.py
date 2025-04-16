import torch
import torch.optim as optim
import numpy as np
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
from vqvae import VQVAE

# Load and preprocess dataset
def load_data(filepath, num_samples=None):
    mat = scipy.io.loadmat(filepath)
    data = mat['HT'].astype('float32')  

    print(f"Loaded data shape from {filepath}: {data.shape}")  # Debugging
    
    # If `num_samples` is provided, use only a subset of the data
    if num_samples is not None:
        data = data[:num_samples]  # Select only the first `num_samples` samples

    # Normalize data (important for stability)
    data = data - 0.5  

    # Fix reshaping from (batch, 1, 2048) → (batch, 2, 32, 32)
    data = np.reshape(data, (data.shape[0], 2, 32, 32))  
    print(f"Final reshaped data shape: {data.shape}")  # Debugging
    return data



# Load training & validation sets
#Subset size (OPTIONAL)
train_subset_size = 5000 # 100000 total
val_subset_size = 1000 # 30000 total

train_data = load_data("data/DATA_Htrainin.mat", train_subset_size)
val_data = load_data("data/DATA_Hvalin.mat", val_subset_size)

# Convert to PyTorch tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32)
val_tensor = torch.tensor(val_data, dtype=torch.float32)

# Create DataLoaders

# modify batch size here
batch_size = 256
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size)

# Set up the model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = VQVAE(input_dim=2).to(device)
#Optimizer is important because it determine VQ loss
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

import time

# Open a log file
log_filename = "training_log.txt"
with open(log_filename, "w") as log_file:
    log_file.write("VQ-VAE Training Log\n")
    log_file.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    log_file.write("=== Training Parameters ===\n")
    log_file.write(f"Learning Rate: {learning_rate}\n")  # Change if different
    log_file.write(f"Batch Size: {batch_size}\n")
    log_file.write(f"Epochs: {num_epochs}\n")
    log_file.write(f"Train Subset Size: {train_subset_size}\n")
    log_file.write(f"Validation Subset Size: {val_subset_size}\n\n")
    log_file.write("=== Training Progress ===\n")
    log_file.write(f"{'Epoch':<10} {'Recon Loss':<15} {'VQ Loss'}\n")
    log_file.write("="*40 + "\n")


# Training loop

#modify number of epochs here
num_epochs = 10 # only 10 because of my smaller subset size, maybe use 50 if using whole training set
for epoch in range(num_epochs):
    model.train()
    total_loss, total_vq_loss = 0, 0

    print("Epoch", epoch+1)
    
    for batch in train_loader:
        data = batch[0].to(device) # should be (batch, 2, 32, 32)
        
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

    # Print loss per epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Recon Loss: {total_loss:.4f}, VQ Loss: {total_vq_loss:.4f}")

    # Append results in table format
    with open(log_filename, "a") as log_file:
        log_file.write(f"{epoch+1:<10} {total_loss:<15.4f} {total_vq_loss:.4f}\n")

# Save the trained model
torch.save(model.state_dict(), "vqvae_model.pth")

# Append final message to log file
with open(log_filename, "a") as log_file:
    log_file.write("="*40 + "\n")
    log_file.write("Training Complete!\n")
    log_file.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")