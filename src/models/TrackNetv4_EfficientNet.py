import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

def rearrange_tensor(input_tensor, order):
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    assert all([dim in order for dim in "BCHWT"]), "Order must contain all of BCHWT"
    return input_tensor.permute(*[order.index(dim) for dim in "BTCHW"])

def reverse_rearrange_tensor(input_tensor, order):
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    return input_tensor.permute(*["BTCHW".index(dim) for dim in order])

def power_normalization(input, a, b):
    return 1 / (1 + torch.exp(-(5 / (0.45 * torch.abs(torch.tanh(a)) + 1e-1)) * (torch.abs(input) - 0.6 * torch.tanh(b))))

class MotionPromptLayer(nn.Module):
    def __init__(self, penalty_weight=0.0):
        super(MotionPromptLayer, self).__init__()
        self.input_permutation = "BTCHW"
        self.input_color_order = "RGB"
        self.color_map = {'R': 0, 'G': 1, 'B': 2}
        self.gray_scale = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)

        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.0))
        
        self.lambda1 = penalty_weight

    def forward(self, video_seq):
        loss = torch.tensor(0.0, device=video_seq.device)

        video_seq = rearrange_tensor(video_seq, self.input_permutation)
        norm_seq = video_seq * 0.225 + 0.45

        idx_list = [self.color_map[idx] for idx in self.input_color_order]
        gray_scale_tensor = self.gray_scale.to(video_seq.device)[idx_list]
        weights = gray_scale_tensor.to(dtype=norm_seq.dtype)
        grayscale_video_seq = torch.einsum("btcwh,c->btwh", norm_seq, weights)

        B, T, H, W = grayscale_video_seq.shape
        frame_diff = grayscale_video_seq[:, 1:] - grayscale_video_seq[:, :-1]

        attention_map = power_normalization(frame_diff, self.a, self.b)
        norm_attention = attention_map.unsqueeze(2)

        if self.training:
            temp_diff = norm_attention[:, 1:] - norm_attention[:, :-1]
            temporal_loss = torch.sum(torch.square(temp_diff)) / (H * W * (T - 2) * B)
            loss = self.lambda1 * temporal_loss

        return attention_map, loss

class FusionLayerTypeA(nn.Module):
    def forward(self, feature_map, attention_map):
        output_1 = feature_map[:, 0, :, :]
        output_2 = feature_map[:, 1, :, :] * attention_map[:, 0, :, :]
        output_3 = feature_map[:, 2, :, :] * attention_map[:, 1, :, :]
        return torch.stack([output_1, output_2, output_3], dim=1)

class FusionLayerTypeB(nn.Module):
    def forward(self, feature_map, attention_map):
        output_1 = feature_map[:, 0, :, :] * attention_map[:, 0, :, :]
        output_2 = feature_map[:, 1, :, :] * ((attention_map[:, 0, :, :] + attention_map[:, 1, :, :])/2)
        output_3 = feature_map[:, 2, :, :] * attention_map[:, 1, :, :]
        return torch.stack([output_1, output_2, output_3], dim=1)


# EfficientNet Building Blocks
class Swish(nn.Module):
    """Swish activation function (SiLU)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, se_channels, 1),
            Swish(),
            nn.Conv2d(se_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution block"""
    def __init__(self, in_channels, out_channels, expand_ratio, stride, 
                 kernel_size=3, se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate
        
        # Expansion phase
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        
        if self.expand:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim, momentum=0.01, eps=1e-3),
                Swish()
            )
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, momentum=0.01, eps=1e-3),
            Swish()
        )
        
        # Squeeze and excitation
        self.se = SqueezeExcitation(hidden_dim, se_ratio)
        
        # Output phase
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
        )
    
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.expand:
            x = self.expand_conv(x)
        
        # Depthwise + SE
        x = self.depthwise(x)
        x = self.se(x)
        
        # Project
        x = self.project(x)
        
        # Residual connection with drop connect
        if self.use_residual:
            if self.training and self.drop_connect_rate > 0:
                # Stochastic depth
                keep_prob = 1 - self.drop_connect_rate
                mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep_prob
                x = x / keep_prob * mask.float()
            x = x + identity
        
        return x


class ConvBNSwish(nn.Module):
    """Standard Conv + BN + Swish block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
        self.act = Swish()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class EfficientNetEncoder(nn.Module):
    def __init__(self, in_channels=9, width_mult=1.0, depth_mult=1.0):
        super().__init__()
        
        def round_channels(channels):
            """Round number of channels to nearest multiple of 8"""
            return int(channels * width_mult // 8) * 8
        
        def round_repeats(repeats):
            """Round number of block repeats"""
            return int(repeats * depth_mult)
        
        # Stem - NO STRIDE to preserve spatial resolution for small objects
        self.stem = ConvBNSwish(in_channels, round_channels(32), 3, stride=1, padding=1)
        
        # EfficientNet-B0 architecture (modified)
        # Format: [expand_ratio, out_channels, num_repeats, stride, kernel_size]
        blocks_args = [
            # Stage 1: 32 -> 16 channels, no downsample
            [1, 16, 1, 1, 3],
            # Stage 2: 16 -> 24 channels, downsample 2x
            [6, 24, 2, 2, 3],
            # Stage 3: 24 -> 40 channels, downsample 2x
            [6, 40, 2, 2, 5],
            # Stage 4: 40 -> 80 channels, downsample 2x
            [6, 80, 3, 2, 3],
            # Stage 5: 80 -> 112 channels, no downsample
            [6, 112, 3, 1, 5],
            # Stage 6: 112 -> 192 channels, downsample 2x (bottleneck)
            [6, 192, 4, 2, 5],
        ]
        
        self.blocks = nn.ModuleList([])
        in_ch = round_channels(32)
        
        # Drop connect rate increases linearly
        total_blocks = sum([round_repeats(b[2]) for b in blocks_args])
        drop_rate_per_block = 0.2 / total_blocks
        block_idx = 0
        
        for expand_ratio, out_ch, num_repeats, stride, kernel_size in blocks_args:
            out_ch = round_channels(out_ch)
            num_repeats = round_repeats(num_repeats)
            
            stage_blocks = []
            for i in range(num_repeats):
                stage_blocks.append(MBConvBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    expand_ratio=expand_ratio,
                    stride=stride if i == 0 else 1,
                    kernel_size=kernel_size,
                    se_ratio=0.25,
                    drop_connect_rate=drop_rate_per_block * block_idx
                ))
                block_idx += 1
            
            self.blocks.append(nn.Sequential(*stage_blocks))
            in_ch = out_ch
        
        # Store output channels for each stage (for skip connections)
        self.out_channels = [
            round_channels(32),   # stem
            round_channels(16),   # stage 1
            round_channels(24),   # stage 2
            round_channels(40),   # stage 3
            round_channels(80),   # stage 4
            round_channels(112),  # stage 5
            round_channels(192),  # stage 6 (bottleneck)
        ]
    
    def forward(self, x):
        """Returns features at different scales for U-Net decoder"""
        features = []
        
        x = self.stem(x)
        features.append(x)  # stem output
        
        for block in self.blocks:
            x = block(x)
            features.append(x)  # stage outputs
        
        return features


class EfficientUNetDecoder(nn.Module):
    """U-Net style decoder with skip connections"""
    def __init__(self, encoder_channels: List[int]):
        super().__init__()
        
        # encoder_channels = [32, 16, 24, 40, 80, 112, 192]
        # We'll upsample from 192 back to original resolution
        
        # Decoder stages (upsample + skip connection + conv blocks)
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6 = nn.Sequential(
            ConvBNSwish(encoder_channels[6] + encoder_channels[5], encoder_channels[5], 3, 1, 1),
            ConvBNSwish(encoder_channels[5], encoder_channels[5], 3, 1, 1)
        )
        
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = nn.Sequential(
            ConvBNSwish(encoder_channels[5] + encoder_channels[4], encoder_channels[4], 3, 1, 1),
            ConvBNSwish(encoder_channels[4], encoder_channels[4], 3, 1, 1)
        )
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Sequential(
            ConvBNSwish(encoder_channels[4] + encoder_channels[3], encoder_channels[3], 3, 1, 1),
            ConvBNSwish(encoder_channels[3], encoder_channels[3], 3, 1, 1)
        )
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Sequential(
            ConvBNSwish(encoder_channels[3] + encoder_channels[2], encoder_channels[2], 3, 1, 1),
            ConvBNSwish(encoder_channels[2], encoder_channels[2], 3, 1, 1)
        )
        
        # Final layers (no more upsampling needed since stem has no stride)
        self.conv2 = nn.Sequential(
            ConvBNSwish(encoder_channels[2] + encoder_channels[1], encoder_channels[1], 3, 1, 1),
            ConvBNSwish(encoder_channels[1], encoder_channels[1], 3, 1, 1)
        )
        
        self.conv1 = nn.Sequential(
            ConvBNSwish(encoder_channels[1] + encoder_channels[0], encoder_channels[0], 3, 1, 1),
            ConvBNSwish(encoder_channels[0], encoder_channels[0], 3, 1, 1)
        )
        
        # Final output
        self.final = nn.Conv2d(encoder_channels[0], 3, 1)
    
    def forward(self, features):
        """
        features: [stem, stage1, stage2, stage3, stage4, stage5, stage6]
        Spatial sizes vary based on stride configuration.
        """
        # Start from bottleneck (stage6)
        x = features[6]
        
        # Helper function to match spatial sizes
        def match_size(x, target):
            if x.shape[2:] != target.shape[2:]:
                x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)
            return x
        
        # Progressively upsample and concatenate skip connections
        x = self.up6(x)
        x = match_size(x, features[5])
        x = torch.cat([x, features[5]], dim=1)
        x = self.conv6(x)
        
        x = self.up5(x)
        x = match_size(x, features[4])
        x = torch.cat([x, features[4]], dim=1)
        x = self.conv5(x)
        
        x = self.up4(x)
        x = match_size(x, features[3])
        x = torch.cat([x, features[3]], dim=1)
        x = self.conv4(x)
        
        x = self.up3(x)
        x = match_size(x, features[2])
        x = torch.cat([x, features[2]], dim=1)
        x = self.conv3(x)
        
        # Match sizes for remaining concatenations
        x = match_size(x, features[1])
        x = torch.cat([x, features[1]], dim=1)
        x = self.conv2(x)
        
        x = match_size(x, features[0])
        x = torch.cat([x, features[0]], dim=1)
        x = self.conv1(x)
        
        # Final 1x1 conv to 3 channels
        x = self.final(x)
        
        return x


class TrackNetV4_EfficientUNet(nn.Module):
    def __init__(self, input_height, input_width, fusion_layer_type="TypeA",
                 width_mult=1.0, depth_mult=1.0):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width

        if fusion_layer_type == "TypeA":
            self.fusion_layer = FusionLayerTypeA()
        elif fusion_layer_type == "TypeB":
            self.fusion_layer = FusionLayerTypeB()
        else:
            raise ValueError("Unknown Motion Fusion Type")

        self.motion_prompt = MotionPromptLayer()

        # EfficientNet encoder
        self.encoder = EfficientNetEncoder(
            in_channels=9,
            width_mult=width_mult,
            depth_mult=depth_mult
        )
        
        # U-Net decoder
        self.decoder = EfficientUNetDecoder(self.encoder.out_channels)
        
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Motion prompt processing
        motion_input = x.view(-1, 3, 3, self.input_height, self.input_width)
        residual_maps, motion_loss = self.motion_prompt(motion_input)

        # Encoder (extract features at multiple scales)
        features = self.encoder(x)
        
        # Decoder (reconstruct with skip connections)
        logits = self.decoder(features)
        
        # Apply motion fusion
        fused = self.fusion_layer(logits, residual_maps)
        
        # Sigmoid activation
        output = self.sigmoid(fused)
        
        return output, motion_loss


# Predefined model variants
def TrackNetV4_EfficientNet_B0(input_height=288, input_width=512, fusion_layer_type="TypeA"):
    """EfficientNet-B0 backbone - balanced speed/accuracy"""
    return TrackNetV4_EfficientUNet(
        input_height, input_width, fusion_layer_type,
        width_mult=1.0, depth_mult=1.0
    )


def TrackNetV4_EfficientNet_Lite(input_height=288, input_width=512, fusion_layer_type="TypeA"):
    """Lightweight variant - faster inference"""
    return TrackNetV4_EfficientUNet(
        input_height, input_width, fusion_layer_type,
        width_mult=0.75, depth_mult=0.75
    )


def TrackNetV4_EfficientNet_B1(input_height=288, input_width=512, fusion_layer_type="TypeA"):
    """EfficientNet-B1 backbone - better accuracy"""
    return TrackNetV4_EfficientUNet(
        input_height, input_width, fusion_layer_type,
        width_mult=1.0, depth_mult=1.1
    )


if __name__ == "__main__":
    # Test the model
    print("Testing TrackNetV4_EfficientUNet variants...\n")
    
    variants = [
        ("EfficientNet-Lite (0.75×)", 0.75, 0.75),
        ("EfficientNet-B0 (1.0×)", 1.0, 1.0),
        ("EfficientNet-B1 (1.1× depth)", 1.0, 1.1),
    ]
    
    for name, width_mult, depth_mult in variants:
        print(f"{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")
        
        model = TrackNetV4_EfficientUNet(
            input_height=288,
            input_width=512,
            fusion_layer_type="TypeA",
            width_mult=width_mult,
            depth_mult=depth_mult
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Test forward pass
        x = torch.randn(2, 9, 288, 512)
        with torch.no_grad():
            output, loss = model(x)
        
        print(f"Parameters:    {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Input shape:   {x.shape}")
        print(f"Output shape:  {output.shape}")
        print(f"Motion loss:   {loss.item():.6f}")
        print(f"Output range:  [{output.min():.3f}, {output.max():.3f}]")
        
        # Verify output shape
        assert output.shape == (2, 3, 288, 512), f"Expected (2, 3, 288, 512), got {output.shape}"
        print("✓ Test passed!\n")
    
    print("="*70)
    print("All variants tested successfully! ✓")
    print("="*70)