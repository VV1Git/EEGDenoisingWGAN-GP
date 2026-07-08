import torch
import torch.nn as nn
from variables import GEN_NUM_LAYERS, DISC_NUM_LAYERS

# --- Discriminator (Critic) for 1D EEG Signals ---
class Discriminator(nn.Module):
    def __init__(self, channels_eeg, seq_len, features_d, num_layers=DISC_NUM_LAYERS):
        super(Discriminator, self).__init__()
        layers = [
            nn.Conv1d(channels_eeg, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_ch = features_d
        for i in range(1, num_layers):
            out_ch = features_d * (2 ** i)
            layers.append(self._block(in_ch, out_ch, 4, 2, 1))
            in_ch = out_ch
        # Final output layer
        final_kernel = seq_len // (2 ** num_layers)
        if final_kernel < 1:
            final_kernel = 1
        layers.append(nn.Conv1d(in_ch, 1, kernel_size=final_kernel, stride=1, padding=0))
        self.disc = nn.Sequential(*layers)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        num_groups = min(8, out_channels)
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.disc(x)

# --- Generator for 1D EEG Denoising with Skip Connections (U-Net) ---
class Generator(nn.Module):
    def __init__(self, channels_eeg, seq_len, features_g, dropout_p=0.2, num_layers=GEN_NUM_LAYERS):
        super(Generator, self).__init__()
        self.num_layers = num_layers
        enc_layers = []
        encoder_channels = []

        # First encoder layer (no norm)
        enc_layers.append(nn.Sequential(
            nn.Conv1d(channels_eeg, features_g, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ))
        encoder_channels.append(features_g)
        in_ch = features_g

        # Remaining encoder layers
        for i in range(1, num_layers):
            out_ch = features_g * (2 ** i)
            enc_layers.append(self._enc_block(in_ch, out_ch, dropout_p))
            encoder_channels.append(out_ch)
            in_ch = out_ch
        self.encoders = nn.ModuleList(enc_layers)
        self.encoder_channels = encoder_channels

        # Decoder (with skip connections)
        dec_layers = []
        decoder_in_channels = []
        decoder_out_channels = []
        # Start from the deepest encoder output
        prev_out_ch = encoder_channels[-1]
        for i in range(num_layers - 1, 0, -1):
            skip_ch = encoder_channels[i - 1]
            out_ch = skip_ch
            in_ch = prev_out_ch + skip_ch
            dec_layers.append(self._dec_block(in_ch, out_ch, dropout_p))
            decoder_in_channels.append(in_ch)
            decoder_out_channels.append(out_ch)
            prev_out_ch = out_ch
        self.decoders = nn.ModuleList(dec_layers)
        # Final layer: input is concat of last decoder output and first encoder output
        self.final = nn.Sequential(
            nn.ConvTranspose1d(encoder_channels[0] * 2, channels_eeg, 4, 2, 1),
            nn.Tanh(),
        )

    def _enc_block(self, in_channels, out_channels, dropout_p):
        num_groups = min(8, out_channels)
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p),
        )

    def _dec_block(self, in_channels, out_channels, dropout_p):
        num_groups = min(8, out_channels)
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        # Encoder
        enc_feats = []
        out = x
        for enc in self.encoders:
            out = enc(out)
            enc_feats.append(out)
        # Decoder with skip connections
        out = enc_feats[-1]
        for i, dec in enumerate(self.decoders):
            skip = enc_feats[-(i + 2)]
            skip = self._crop_or_pad(skip, out.size(-1))
            out = dec(torch.cat([out, skip], dim=1))
        # Final skip connection
        skip0 = self._crop_or_pad(enc_feats[0], out.size(-1))
        out = self.final(torch.cat([out, skip0], dim=1))
        return out

    def _crop_or_pad(self, tensor, target_length):
        """Crop or pad tensor along the last dimension to match target_length.

        NOTE: this is load-bearing, not dead code. At seq_len=512 the crop
        branch fires on every decoder skip (halving each skip length before the
        concat), so removing it would change outputs and break the cat. The
        crop branch is ordered first because it is the case that always fires.
        """
        current_length = tensor.size(-1)
        if current_length > target_length:  # hot path (always fires at 512)
            return tensor[..., :target_length]
        if current_length == target_length:
            return tensor
        pad_amt = target_length - current_length
        # Pad at the end
        return nn.functional.pad(tensor, (0, pad_amt))

def initialize_weights(model):
    # Same initial values as before (conv/convT weights ~ N(0, 0.02), biases 0,
    # GroupNorm weight 1 / bias 0). Uses .weight/.bias instead of the deprecated
    # .data and an elif so GroupNorm's bias isn't re-zeroed by a trailing branch.
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

# Example for testing the model shapes (optional)
if __name__ == "__main__":
    # Test Discriminator
    batch_size = 16
    channels = 1
    seq_len = 512
    features_d = 16
    features_g = 16
    x = torch.randn(batch_size, channels, seq_len)
    disc = Discriminator(channels, seq_len, features_d)
    gen = Generator(channels, seq_len, features_g)
    # This is only a shape smoke test; skip building the autograd graph.
    with torch.inference_mode():
        out_disc = disc(x)
        print(f"Discriminator input shape: {x.shape}")
        print(f"Discriminator output shape: {out_disc.shape}") # Should be (batch_size, 1, 1)

        # Test Generator
        out_gen = gen(x) # Generator takes noisy EEG as input
        print(f"Generator input shape: {x.shape}")
        print(f"Generator output shape: {out_gen.shape}") # Should be (batch_size, channels, seq_len)

    # Test weight initialization
    initialize_weights(gen)
    initialize_weights(disc)
    print("\nWeights initialized for Generator and Discriminator.")
