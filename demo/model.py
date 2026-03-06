import torch
import torch.nn as nn

# Model configuration
GEN_NUM_LAYERS = 6

class Generator(nn.Module):
    def __init__(self, channels_eeg=1, seq_len=512, features_g=32, dropout_p=0.2, num_layers=GEN_NUM_LAYERS):
        super(Generator, self).__init__()
        self.num_layers = num_layers
        enc_layers = []
        encoder_channels = []
        
        # First encoder layer (no norm)
        enc_layers.append(nn.Sequential(
            nn.Conv1d(channels_eeg, features_g, 4, 2, 1),
            nn.LeakyReLU(0.2),
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
        prev_out_ch = encoder_channels[-1]
        for i in range(num_layers - 1, 0, -1):
            skip_ch = encoder_channels[i - 1]
            out_ch = skip_ch
            in_ch = prev_out_ch + skip_ch
            dec_layers.append(self._dec_block(in_ch, out_ch, dropout_p))
            prev_out_ch = out_ch
        
        self.decoders = nn.ModuleList(dec_layers)
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose1d(encoder_channels[0] * 2, channels_eeg, 4, 2, 1),
            nn.Tanh(),
        )
    
    def _enc_block(self, in_channels, out_channels, dropout_p):
        num_groups = min(8, out_channels)
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, affine=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_p),
        )
    
    def _dec_block(self, in_channels, out_channels, dropout_p):
        num_groups = min(8, out_channels)
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, affine=True),
            nn.ReLU(),
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
        current_length = tensor.size(-1)
        if current_length == target_length:
            return tensor
        elif current_length > target_length:
            return tensor[..., :target_length]
        else:
            pad_amt = target_length - current_length
            return nn.functional.pad(tensor, (0, pad_amt))
