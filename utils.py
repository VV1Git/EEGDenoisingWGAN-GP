import torch

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
    # Allocate directly on the target device (avoids a per-call H2D copy on GPU;
    # on CPU device="cpu" this is identical to the previous CPU allocation).
    alpha = torch.rand((BATCH_SIZE, 1, 1), device=device)

    # The gradient penalty involves a create_graph=True double-backward, which
    # is numerically unsafe under fp16 autocast (the squared-norm penalty
    # overflows/underflows and the (norm-1)^2 target becomes meaningless). Force
    # the whole GP computation to run in fp32 regardless of any enclosing
    # autocast. On CPU (or when autocast is not active) this is a no-op, so the
    # CPU numerics are byte-identical to before.
    real = real.float()
    fake = fake.float()

    # Create interpolated samples
    # interpolated_images = real * alpha + fake * (1 - alpha)
    # The repeat operation is not needed if alpha is already shaped correctly for broadcasting
    interpolated_signals = real * alpha + fake * (1 - alpha)

    # Ensure gradients can be computed for interpolated_signals
    interpolated_signals.requires_grad_(True)

    # Disable autocast so the critic forward + autograd.grad run in fp32 even if
    # the caller is inside a torch.autocast(...) region. autocast_device_type is
    # "cuda" for a GPU device and "cpu" otherwise; enabled=False makes this a
    # no-op on the CPU path.
    autocast_device_type = "cuda" if (isinstance(device, str) and device != "cpu") or \
        (hasattr(device, "type") and device.type == "cuda") else "cpu"
    with torch.autocast(device_type=autocast_device_type, enabled=False):
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

