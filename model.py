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
        # Encoder (Contracting Path)
        self.enc1 = nn.Sequential(
            nn.Conv1d(channels_eeg, features_g, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(features_g),
            nn.LeakyReLU(0.2)
        )  # -> features_g x L/2

        self.enc2 = nn.Sequential(
            nn.Conv1d(features_g, features_g * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(features_g * 2),
            nn.LeakyReLU(0.2)
        )  # -> features_g*2 x L/4

        self.enc3 = nn.Sequential(
            nn.Conv1d(features_g * 2, features_g * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(features_g * 4),
            nn.LeakyReLU(0.2)
        )  # -> features_g*4 x L/8

        self.enc4 = nn.Sequential(
            nn.Conv1d(features_g * 4, features_g * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(features_g * 8),
            nn.LeakyReLU(0.2)
        )  # -> features_g*8 x L/16

        # Bottleneck (no downsampling)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(features_g * 8, features_g * 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(features_g * 16),
            nn.LeakyReLU(0.2)
        )  # -> features_g*16 x L/16

        # Decoder (Expansive Path)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(features_g * 8),
            nn.LeakyReLU(0.2)
        )  # -> features_g*8 x L/8

        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(features_g * 16, features_g * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(features_g * 4),
            nn.LeakyReLU(0.2)
        )  # -> features_g*4 x L/4

        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(features_g * 8, features_g * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(features_g * 2),
            nn.LeakyReLU(0.2)
        )  # -> features_g*2 x L/2

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(features_g * 4, features_g, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(features_g),
            nn.LeakyReLU(0.2)
        )  # -> features_g x L

        # Output layer
        self.final = nn.Sequential(
            nn.ConvTranspose1d(features_g * 2, channels_eeg, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )  # -> channels_eeg x L

    def _crop_to_match(self, enc_feat, dec_feat):
        # Crop enc_feat along the last dimension to match dec_feat's length
        if enc_feat.shape[-1] > dec_feat.shape[-1]:
            diff = enc_feat.shape[-1] - dec_feat.shape[-1]
            enc_feat = enc_feat[..., diff // 2 : enc_feat.shape[-1] - (diff - diff // 2)]
        elif enc_feat.shape[-1] < dec_feat.shape[-1]:
            # Optional: pad if encoder feature is smaller (shouldn't happen in this architecture)
            pad = dec_feat.shape[-1] - enc_feat.shape[-1]
            enc_feat = nn.functional.pad(enc_feat, (pad // 2, pad - pad // 2))
        return enc_feat

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # features_g x L/2
        e2 = self.enc2(e1) # features_g*2 x L/4
        e3 = self.enc3(e2) # features_g*4 x L/8
        e4 = self.enc4(e3) # features_g*8 x L/16

        # Bottleneck
        b = self.bottleneck(e4) # features_g*16 x L/16

        # Decoder with skip connections (concatenate along channel dim)
        d4 = self.dec4(b)                   # features_g*8 x L/8
        e4_cropped = self._crop_to_match(e4, d4)
        d4 = torch.cat([d4, e4_cropped], dim=1)     # features_g*16 x L/8

        d3 = self.dec3(d4)                  # features_g*4 x L/4
        e3_cropped = self._crop_to_match(e3, d3)
        d3 = torch.cat([d3, e3_cropped], dim=1)     # features_g*8 x L/4

        d2 = self.dec2(d3)                  # features_g*2 x L/2
        e2_cropped = self._crop_to_match(e2, d2)
        d2 = torch.cat([d2, e2_cropped], dim=1)     # features_g*4 x L/2

        d1 = self.dec1(d2)                  # features_g x L
        e1_cropped = self._crop_to_match(e1, d1)
        d1 = torch.cat([d1, e1_cropped], dim=1)     # features_g*2 x L

        out = self.final(d1)                # channels_eeg x L
        return out


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
