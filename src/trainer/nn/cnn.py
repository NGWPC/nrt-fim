import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

class DecoderBlock(nn.Module):
    """
    Decoder block for the expansive path, implementing Eq. (2) from the paper
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.swish1 = Swish()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.swish2 = Swish()
        
    def forward(self, x, skip):
        x_up = self.upsample(x)
        
        if x_up.size()[2:] != skip.size()[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([x_up, skip], dim=1)
        
        x = self.swish1(self.bn1(self.conv1(x)))
        x = self.swish2(self.bn2(self.conv2(x)))
        
        return x


class FModel(nn.Module):
    """
    U-Net architecture with EfficientNet-B1 encoder as described in the paper
    """
    def __init__(self, num_classes=1):
        super().__init__()
        
        self.encoder = efficientnet_b1()
        
        self.initial_conv = nn.Sequential(
            self.encoder.features[0],
        )
        
        self.encoder_blocks = nn.ModuleList()
        
        self.encoder_blocks.append(nn.Sequential(
            self.encoder.features[1],
        ))
        
        self.encoder_blocks.append(nn.Sequential(
            self.encoder.features[2], 
            self.encoder.features[3],
        ))
        
        self.encoder_blocks.append(nn.Sequential(
            self.encoder.features[4],
            self.encoder.features[5],
        ))
        
        self.encoder_blocks.append(nn.Sequential(
            self.encoder.features[6], 
            self.encoder.features[7],
        ))
        
        self.encoder_blocks.append(nn.Sequential(
            self.encoder.features[8], 
            *[self.encoder.features[i] for i in range(9, 16)]
        ))
        
        self.skip_channels = [32, 24, 40, 80, 112]
        
        self.bottleneck = nn.Sequential(
            *[self.encoder.features[i] for i in range(16, len(self.encoder.features))]
        )
        
        self.decoder_channels = [320, 112, 80, 40, 24]
        
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(self.skip_channels)):
            if i == 0:
                self.decoder_blocks.append(DecoderBlock(
                    self.decoder_channels[0], 
                    self.skip_channels[4], 
                    self.decoder_channels[1]
                ))
            else:
                self.decoder_blocks.append(DecoderBlock(
                    self.decoder_channels[i], 
                    self.skip_channels[4-i], 
                    self.decoder_channels[i+1] if i < len(self.decoder_channels)-1 else 32
                ))
            
        # PSWApp = Clip(C`, 0, 1)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        skip_features = []
        
        x = self.initial_conv(x)
        
        for block in self.encoder_blocks:
            x = block(x)
            skip_features.append(x)
        
        x = self.bottleneck(x)
        
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, skip_features[4-i]) 
            
        x = self.final_conv(x)
        x = self.sigmoid(x) 
        
        return x
    