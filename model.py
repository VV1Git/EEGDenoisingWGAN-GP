import torch
import torch.nn as nn

# --- Discriminator (Critic) for 1D EEG Signals ---
class Discriminator(nn.Module):
    def __init__(self, channels_eeg, seq_len, features_d):
        super(Discriminator, self).__init__()
        # Input: N x channels_eeg x seq_len (e.g., N x 1 x 512)
        self.disc = nn.Sequential(
            # First layer: Reduce sequence length, increase features
            # Example: 1x512 -> features_d x 256 (kernel_size=4, stride=2, padding=1)
            nn.Conv1d(channels_eeg, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),   # e.g., 256 -> 128
            self._block(features_d * 2, features_d * 4, 4, 2, 1), # e.g., 128 -> 64
            self._block(features_d * 4, features_d * 8, 4, 2, 1), # e.g., 64 -> 32
            # After these blocks, the sequence length will be significantly reduced.
            # Calculate output sequence length after blocks:
            # L_out = floor((L_in + 2*padding - kernel_size)/stride) + 1
            # For seq_len=512, kernel=4, stride=2, padding=1:
            # L1 = floor((512 + 2*1 - 4)/2) + 1 = 256
            # L2 = floor((256 + 2*1 - 4)/2) + 1 = 128
            # L3 = floor((128 + 2*1 - 4)/2) + 1 = 64
            # L4 = floor((64 + 2*1 - 4)/2) + 1 = 32
            # The final Conv1d should map the last feature map to a single scalar per batch item.
            # This requires a kernel size that matches the final sequence length.
            nn.Conv1d(features_d * 8, 1, kernel_size=32, stride=1, padding=0), # Output: N x 1 x 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False, # Bias is often set to False when using InstanceNorm
            ),
            nn.InstanceNorm1d(out_channels, affine=True), # Use InstanceNorm1d for 1D data
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


# --- Generator for 1D EEG Denoising ---
class Generator(nn.Module):
    def __init__(self, channels_eeg, seq_len, features_g):
        super(Generator, self).__init__()
        # Input: N x channels_eeg x seq_len (noisy EEG, e.g., N x 1 x 512)
        # Output: N x channels_eeg x seq_len (denoised EEG, e.g., N x 1 x 512)
        # This architecture is a U-Net like structure or an encoder-decoder
        # for denoising, rather than generating from noise.
        # We'll use Conv1d for downsampling and ConvTranspose1d for upsampling.

        # Encoder (Downsampling path)
        self.encoder = nn.Sequential(
            # Input: N x 1 x 512
            nn.Conv1d(channels_eeg, features_g, kernel_size=4, stride=2, padding=1), # -> N x features_g x 256
            nn.LeakyReLU(0.2),
            self._enc_block(features_g, features_g * 2, 4, 2, 1), # -> N x features_g*2 x 128
            self._enc_block(features_g * 2, features_g * 4, 4, 2, 1), # -> N x features_g*4 x 64
            self._enc_block(features_g * 4, features_g * 8, 4, 2, 1), # -> N x features_g*8 x 32
            self._enc_block(features_g * 8, features_g * 16, 4, 2, 1), # -> N x features_g*16 x 16
        )

        # Decoder (Upsampling path)
        self.decoder = nn.Sequential(
            # Input from encoder: N x features_g*16 x 16
            self._dec_block(features_g * 16, features_g * 8, 4, 2, 1), # -> N x features_g*8 x 32
            self._dec_block(features_g * 8, features_g * 4, 4, 2, 1), # -> N x features_g*4 x 64
            self._dec_block(features_g * 4, features_g * 2, 4, 2, 1), # -> N x features_g*2 x 128
            self._dec_block(features_g * 2, features_g, 4, 2, 1), # -> N x features_g x 256
            # Final layer to output the original channel count
            nn.ConvTranspose1d(features_g, channels_eeg, kernel_size=4, stride=2, padding=1), # -> N x channels_eeg x 512
            nn.Tanh(), # Output values typically in [-1, 1] if input was normalized
        )

    def _enc_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channels), # Use BatchNorm1d for 1D data
            nn.LeakyReLU(0.2),
        )

    def _dec_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channels), # Use BatchNorm1d for 1D data
            nn.ReLU(),
        )

    def forward(self, x):
        # For a simple encoder-decoder, just pass through.
        # For U-Net, you'd add skip connections here.
        encoded = self.encoder(x)
        denoised = self.decoder(encoded)
        return denoised


def initialize_weights(model):
    # Initializes weights for 1D convolutional layers
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d, nn.InstanceNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        # No need to initialize bias if bias=False in Conv layers
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# Example for testing the model shapes (optional)
if __name__ == "__main__":
    # Test Discriminator
    batch_size = 16
    channels = 1
    seq_len = 512
    features_d = 16
    x = torch.randn(batch_size, channels, seq_len)
    disc = Discriminator(channels, seq_len, features_d)
    out_disc = disc(x)
    print(f"Discriminator input shape: {x.shape}")
    print(f"Discriminator output shape: {out_disc.shape}") # Should be (batch_size, 1, 1)

    # Test Generator
    features_g = 16
    gen = Generator(channels, seq_len, features_g)
    out_gen = gen(x) # Generator takes noisy EEG as input
    print(f"Generator input shape: {x.shape}")
    print(f"Generator output shape: {out_gen.shape}") # Should be (batch_size, channels, seq_len)

    # Test weight initialization
    initialize_weights(gen)
    initialize_weights(disc)
    print("\nWeights initialized for Generator and Discriminator.")
