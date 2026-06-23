"""Explainable AI tools for SARResNet, featuring Grad-CAM."""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Standard Grad-CAM implementation using PyTorch forward/backward hooks.
    Extracts spatial attribution maps to interpret model predictions.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register PyTorch hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple, we take the first element
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generates a normalized Grad-CAM heatmap for a given input tensor.
        
        Args:
            input_tensor: Shape (1, C, H, W)
            
        Returns:
            A 2D numpy array of the same HxW shape, normalized between 0 and 1.
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        output.backward(retain_graph=True)

        # 1. Get the globally averaged gradients (weights for each channel)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)

        # 2. Multiply each activation channel by its corresponding gradient weight
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        # 3. Average across channels to get the raw heatmap
        heatmap = torch.mean(activations, dim=0).squeeze()

        # 4. Apply ReLU 
        heatmap = F.relu(heatmap)

        # 5. Normalize between 0 and 1
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)

        # 6. Resize the heatmap to match the original input spatial dimensions
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        heatmap = F.interpolate(
            heatmap, 
            size=(input_tensor.size(2), input_tensor.size(3)), 
            mode='bilinear', 
            align_corners=False
        )

        return heatmap.squeeze().detach().cpu().numpy()


def generate_heatmap(input_tensor: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    """
    Convenience function that automatically attaches to the final convolutional 
    block of the SARResNet backbone.
    """
    # For ResNet (18 or 50), the final block is layer4.
    target_layer = model.net.layer4[-1]
    
    grad_cam = GradCAM(model, target_layer)
    return grad_cam.generate_heatmap(input_tensor)