import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple CNN architecture equivalent to the structure in the C implementation:
    Conv2D -> MaxPool2D
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(SimpleCNN, self).__init__()
        
        # Convolution Layer:
        # - Kernel size: 3x3 (same as Base_conv_mp in C code)
        # - Stride: 1
        # - Padding: 0 (no zero-padding)
        # - No bias (for structural similarity with hardware implementation)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False
        )
        
        # Max Pooling Layer:
        # - Pool size: 2x2
        # - Stride: 2
        # - Reduces the spatial size by half after convolution
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        """
        Forward pass:
        1. Apply convolution
        2. Apply max pooling
        """
        x = self.conv(x)    # Step 1: Convolution operation
        x = self.pool(x)    # Step 2: Max pooling operation
        return x


# ===== Example usage =====
if __name__ == "__main__":
    # Network parameters
    inside = 4      # Input height/width
    c_in = 1024     # Number of input channels
    c_out = 4       # Number of output channels
    
    # Create model instance
    model = SimpleCNN(in_channels=c_in, out_channels=c_out)
    
    # Create dummy input tensor with the same shape as in C code
    # Shape: batch_size=1, channels=c_in, height=inside, width=inside
    input_tensor = torch.arange(
        1, inside * inside * c_in + 1, dtype=torch.float32
    ).reshape(1, c_in, inside, inside)
    
    # Run forward pass
    output_tensor = model(input_tensor)
    
    print("Input shape :", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    # Output height/width = (inside - 2) // 2 due to Conv(3x3, no padding) + Pool(2x2)
