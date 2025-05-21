import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import initialization as init

# Todo: Define the SEBlock class
class SEBlock(nn.Module):
    # Squeeze-and-Excitation block for channel attention.
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    # Forward function
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Todo: Define the Encoding class
class PositionalEncoding(nn.Module):
    # Positional encoding for transformer inputs.
    def __init__(self, d_model, max_len=2500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine and cosine functions to even and odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# Todo: Define the TransformerBottleneck class
class TransformerBottleneck(nn.Module):
    # Transformer bottleneck to process encoder output.
    def __init__(self, in_channels, num_heads=8, num_layers=4):
        super().__init__()
        self.pos_encoder = PositionalEncoding(in_channels)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_channels, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )

    """
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        # Apply positional encoding
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Reshape back to [B, C, H, W]
        x = x.permute(1, 2, 0).reshape(B, C, H, W)
        return x
    """
    def forward(self, x):
        B, C, H, W = x.shape
        downsample_factor = 2
        x = F.adaptive_avg_pool2d(x, output_size=(H // downsample_factor, W // downsample_factor))  # or (H//2, W//2)

        H_ds, W_ds = x.shape[2], x.shape[3]  # keep new dimensions
        x = x.flatten(2).permute(0, 2, 1)  # [B, H_ds * W_ds, C]
        x = self.pos_encoder(x)
        x = self.transformer(x)

        x = x.permute(0, 2, 1).contiguous().view(B, C, H_ds, W_ds)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x


# Todo: Define the DecoderBlockSE class
class DecoderBlockSE(nn.Module):
    # Decoder block with Squeeze-and-Excitation.
    def __init__(self, in_channels, skip_channels, out_channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels, reduction)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            """
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            """
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.se(x)
        return x

# Todo: Define the SEUNetTrans class
# This will be the main model class that combines the encoder, bottleneck, and decoder.
class SEUNetTrans(nn.Module):
    # SE-UNet with Transformer bottleneck.
    def __init__(self, encoder_name='resnet18', in_channels=3, classes=3, encoder_weights='imagenet'):
        super().__init__()
        
        # Encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )
        encoder_channels = self.encoder.out_channels
        
        # Bottleneck with Transformer
        # The bottleneck processes the output of the encoder before passing it to the decoder.
        self.bottleneck = TransformerBottleneck(encoder_channels[-1])
        
        # Decoder
        decoder_channels = (256, 128, 64, 32, 16)
        skip_channels = encoder_channels[:-1][::-1]  # Reverse excluding bottleneck
        in_channels = [encoder_channels[-1]] + list(decoder_channels[:-1])
        
        self.decoder = nn.ModuleList([
            DecoderBlockSE(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, decoder_channels)
        ])
        
        # The final segmentation head to produce the output.
        self.seg_head = nn.Conv2d(decoder_channels[-1], classes, kernel_size=3, padding=1)
        
        # Initialize weights
        self.apply(init.initialize_decoder)
        
    # defines how input image → output segmentation map    
    def forward(self, x):
        # Encoder
        features = self.encoder(x) # CNN backbone
        # features: list of feature maps from encoder layers
        # The last feature map is the bottleneck input.
        skips = features[:-1][::-1]  # Reverse skips excluding bottleneck
        
        # Bottleneck
        # x = self.bottleneck(features[-1])
        x = features[-1]
        res = x  # Save original
        x = self.bottleneck(x)
        if x.shape == res.shape:
            x = x + res
        
        # Decoder
        for i, decoder_block in enumerate(self.decoder):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        
        # Segmentation map
        x = self.seg_head(x)
        return x