"""
Authors: Iker Aldasoro
         Jon Lejardi

Date: 16/12/2024
"""

import torch


def get_cuda():
    """Gets the available device

    Returns:
        device: Device used
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nGPU Information:")
        print("- GPU Available: Yes")
        print(f"- GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"- Number of GPUs: {torch.cuda.device_count()}")
        print(f"- CUDA Version: {torch.version.cuda}")

        # Print memory information
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        memory_cached = torch.cuda.memory_reserved(0) / 1024**2
        print(f"- GPU Memory Allocated: {memory_allocated:.2f} MB")
        print(f"- GPU Memory Cached: {memory_cached:.2f} MB")
    else:
        device = torch.device("cpu")
        print("\nGPU not available, using CPU instead")

    return device
