import torch

print("CUDA Available:", torch.cuda.is_available())  # Should print: True
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))  # Shows GPU Model
    print("CUDA Version:", torch.version.cuda)  # Should match your installed CUDA version
    print("PyTorch CUDA Version:", torch.backends.cudnn.version())  # Checks cuDNN version
print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
print(f"Current GPU index: {torch.cuda.current_device()}")