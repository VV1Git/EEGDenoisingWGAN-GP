import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device="cpu"):
    """
    Calculates the gradient penalty for WGAN-GP.
    Adapted for 1D time-series data.

    Args:
        critic (nn.Module): The critic (discriminator) model.
        real (torch.Tensor): Batch of real (clean) EEG signals. Shape: (N, C, L)
        fake (torch.Tensor): Batch of fake (denoised) EEG signals. Shape: (N, C, L)
        device (str): The device ('cpu' or 'cuda') to perform calculations on.

    Returns:
        torch.Tensor: The calculated gradient penalty.
    """
    # Get batch size, channels, and sequence length
    BATCH_SIZE, C, L = real.shape

    # Generate random interpolation weights (alpha)
    # Shape: (BATCH_SIZE, 1, 1) to broadcast correctly over (C, L)
    alpha = torch.rand((BATCH_SIZE, 1, 1)).to(device)

    # Create interpolated samples
    # interpolated_images = real * alpha + fake * (1 - alpha)
    # The repeat operation is not needed if alpha is already shaped correctly for broadcasting
    interpolated_signals = real * alpha + fake * (1 - alpha)
    
    # Ensure gradients can be computed for interpolated_signals
    interpolated_signals.requires_grad_(True)

    # Calculate critic scores for interpolated samples
    mixed_scores = critic(interpolated_signals)

    # Take the gradient of the scores with respect to the interpolated signals
    gradient = torch.autograd.grad(
        inputs=interpolated_signals,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), # Dummy gradients to backpropagate
        create_graph=True, # Required to compute second-order gradients for GP
        retain_graph=True, # Required if graph is needed for subsequent backward calls (e.g., generator update)
    )[0] # [0] because autograd.grad returns a tuple of gradients for each input

    # Flatten the gradients: (BATCH_SIZE, C, L) -> (BATCH_SIZE, C*L)
    gradient = gradient.view(gradient.shape[0], -1)

    # Calculate the L2 norm (magnitude) of the gradients for each sample in the batch
    gradient_norm = gradient.norm(2, dim=1)

    # Calculate the gradient penalty: (norm - 1)^2, then mean over the batch
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(state, filename="eeg_wgan_gp_checkpoint.pth.tar"):
    """
    Saves the model and optimizer states to a checkpoint file.
    """
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gen, critic): # Renamed 'disc' to 'critic' for consistency
    """
    Loads model and optimizer states from a checkpoint file.
    """
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    critic.load_state_dict(checkpoint['critic']) # Use 'critic' key
    # You might also want to load optimizer states if resuming training
    # opt_gen.load_state_dict(checkpoint['opt_gen'])
    # opt_critic.load_state_dict(checkpoint['opt_critic'])

