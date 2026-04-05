import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

"""
TrackNetV4 with Modified CSPNeXt Backbone - Optimized for Ball Tracking

Key modifications from standard CSPNeXt:
1. NO stride in stem - preserve spatial resolution
2. Use standard convolutions (not depthwise) in early stages
3. Higher BatchNorm momentum for stable training
4. Optional: Remove channel attention in early stages
5. Shallower network with less aggressive downsampling
"""

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
    return 1 / (1 + torch.exp(-(5 / (0.45 * torch.abs(torch.tanh(a)) +1e-5)) * (torch.abs(input) - 0.6 * torch.tanh(b))))

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


# Modified components for ball tracking
class ConvModule(nn.Module):
    """Conv2d + BatchNorm + Activation with configurable momentum"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 bn_momentum=0.1, activation='silu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum, eps=1e-5)
        
        if activation == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class StandardConvBlock(nn.Module):
    """Standard convolution block - better for small objects than depthwise"""
    def __init__(self, in_channels, out_channels, expand_ratio=0.5, 
                 add_identity=True, bn_momentum=0.1):
        super().__init__()
        hidden_channels = int(out_channels * expand_ratio)
        self.conv1 = ConvModule(in_channels, hidden_channels, 1, 1, 0, bn_momentum)
        self.conv2 = ConvModule(hidden_channels, hidden_channels, 3, 1, 1, bn_momentum)
        self.conv3 = ConvModule(hidden_channels, out_channels, 1, 1, 0, bn_momentum)
        self.add_identity = add_identity and in_channels == out_channels
    
    def forward(self, x):
        y = self.conv3(self.conv2(self.conv1(x)))
        if self.add_identity:
            return x + y
        return y


class CSPLayerBallTracking(nn.Module):
    """CSP Layer optimized for ball tracking - no channel attention in early stages"""
    def __init__(self, in_channels, out_channels, num_blocks=1, 
                 expand_ratio=0.5, add_identity=True, 
                 use_standard_conv=True, channel_attention=False,
                 bn_momentum=0.1):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        
        self.main_conv = ConvModule(in_channels, mid_channels, 1, 1, 0, bn_momentum)
        self.short_conv = ConvModule(in_channels, mid_channels, 1, 1, 0, bn_momentum)
        self.final_conv = ConvModule(2 * mid_channels, out_channels, 1, 1, 0, bn_momentum)
        
        if use_standard_conv:
            self.blocks = nn.Sequential(*[
                StandardConvBlock(mid_channels, mid_channels, 1.0, add_identity, bn_momentum) 
                for _ in range(num_blocks)
            ])
        else:
            # Use depthwise only in deeper layers
            self.blocks = nn.Sequential(*[
                CSPNeXtBlock(mid_channels, mid_channels, 1.0, add_identity, bn_momentum) 
                for _ in range(num_blocks)
            ])
        
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)
    
    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        x_final = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class CSPNeXtBlock(nn.Module):
    """CSPNeXt block with depthwise convolutions - for deeper layers only"""
    def __init__(self, in_channels, out_channels, expand_ratio=0.5, 
                 add_identity=True, bn_momentum=0.1):
        super().__init__()
        hidden_channels = int(out_channels * expand_ratio)
        self.conv1 = ConvModule(in_channels, hidden_channels, 3, 1, 1, bn_momentum)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 5, 1, 2, 
                     groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels, momentum=bn_momentum, eps=1e-5),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum, eps=1e-5),
            nn.SiLU(inplace=True)
        )
        self.add_identity = add_identity and in_channels == out_channels
    
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.add_identity:
            return x + y
        return y


class ChannelAttention(nn.Module):
    """Channel attention - use sparingly for small object tracking"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)
    
    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling - only in bottleneck"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), bn_momentum=0.1):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid_channels, 1, 1, 0, bn_momentum)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(conv2_channels, out_channels, 1, 1, 0, bn_momentum)

    def forward(self, x):
        x = self.conv1(x)
        with torch.amp.autocast('cuda', enabled=False):
            x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


class CSPNeXtBackbone_BallTracking(nn.Module):
    """
    Modified CSPNeXt backbone optimized for ball tracking.
    
    Key differences from standard CSPNeXt:
    1. NO stride in stem - preserves spatial resolution
    2. Only 3 downsampling stages (not 4) - less aggressive
    3. Standard convolutions in early stages (not depthwise)
    4. No channel attention in early stages
    5. Higher BatchNorm momentum (0.1 vs 0.03)
    """
    def __init__(self, in_channels=9, deepen_factor=1.0, widen_factor=1.0,
                 expand_ratio=0.5, bn_momentum=0.1):
        super().__init__()
        
        # Modified architecture: [in_ch, out_ch, num_blocks, add_identity, use_spp, use_standard_conv, channel_attn]
        # Less aggressive downsampling, preserve spatial resolution
        arch_settings = [
            [64, 128, 3, True, False, True, False],   # Stage 1: standard conv, no attention
            [128, 256, 6, True, False, True, False],  # Stage 2: standard conv, no attention
            [256, 512, 6, True, False, False, True],  # Stage 3: depthwise ok, with attention
            [512, 512, 3, False, True, False, True],  # Stage 4: bottleneck with SPP
        ]
        
        # Stem - NO STRIDE! Critical for small objects
        base_channels = int(arch_settings[0][0] * widen_factor)
        self.stem = nn.Sequential(
            ConvModule(in_channels, base_channels // 2, 3, stride=1, padding=1, bn_momentum=bn_momentum),
            ConvModule(base_channels // 2, base_channels // 2, 3, stride=1, padding=1, bn_momentum=bn_momentum),
            ConvModule(base_channels // 2, base_channels, 3, stride=1, padding=1, bn_momentum=bn_momentum)
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        for i, (in_ch, out_ch, num_blocks, add_identity, use_spp, 
                use_standard_conv, channel_attn) in enumerate(arch_settings):
            in_ch = int(in_ch * widen_factor)
            out_ch = int(out_ch * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            
            stage = []
            # Downsample conv
            stage.append(ConvModule(in_ch, out_ch, 3, stride=2, padding=1, bn_momentum=bn_momentum))
            
            # SPP if needed
            if use_spp:
                stage.append(SPPBottleneck(out_ch, out_ch, kernel_sizes=(5, 9, 13), 
                                          bn_momentum=bn_momentum))
            
            # CSP Layer
            stage.append(CSPLayerBallTracking(
                out_ch, out_ch, 
                num_blocks=num_blocks,
                add_identity=add_identity,
                expand_ratio=expand_ratio,
                use_standard_conv=use_standard_conv,
                channel_attention=channel_attn,
                bn_momentum=bn_momentum
            ))
            
            self.stages.append(nn.Sequential(*stage))
    
    def forward(self, x):
        """Returns features at different scales for decoder."""
        x = self.stem(x)  # No downsampling!
        
        outputs = [x]
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        
        # Return features: [stem_out, stage1_out, stage2_out, stage3_out, stage4_out]
        return outputs


class TrackNetV4_CSPNeXt_BallTracking(nn.Module):
    """
    TrackNetV4 with Modified CSPNeXt backbone optimized for ball tracking.
    
    Improvements over standard CSPNeXt:
    1. Preserved spatial resolution in early layers
    2. Standard convolutions for better small object detection
    3. Less aggressive downsampling (4x instead of 32x)
    4. Stable BatchNorm settings
    5. Selective use of channel attention
    """
    def __init__(self, input_height, input_width, fusion_layer_type="TypeA",
                 deepen_factor=1.0, widen_factor=1.0, bn_momentum=0.1):
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

        # Modified CSPNeXt Backbone (encoder)
        self.backbone = CSPNeXtBackbone_BallTracking(
            in_channels=9,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            expand_ratio=0.5,
            bn_momentum=bn_momentum
        )
        
        # Calculate actual channel dimensions
        c1 = int(64 * widen_factor)   # stem output
        c2 = int(128 * widen_factor)  # stage1 output
        c3 = int(256 * widen_factor)  # stage2 output
        c4 = int(512 * widen_factor)  # stage3 output
        c5 = int(512 * widen_factor)  # stage4 output (bottleneck)
        
        # Decoder - only 4 upsampling stages needed (stem has no stride)
        # Upsample 1: from bottleneck (c5) + skip from stage3 (c4)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv11 = ConvModule(c5 + c4, c4, 3, 1, 1, bn_momentum)
        self.conv12 = ConvModule(c4, c4, 3, 1, 1, bn_momentum)
        self.conv13 = ConvModule(c4, c4, 3, 1, 1, bn_momentum)

        # Upsample 2: from conv13 (c4) + skip from stage2 (c3)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv14 = ConvModule(c4 + c3, c3, 3, 1, 1, bn_momentum)
        self.conv15 = ConvModule(c3, c3, 3, 1, 1, bn_momentum)

        # Upsample 3: from conv15 (c3) + skip from stage1 (c2)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv16 = ConvModule(c3 + c2, c2, 3, 1, 1, bn_momentum)
        self.conv17 = ConvModule(c2, c2, 3, 1, 1, bn_momentum)
        
        # Upsample 4: from conv17 (c2) + skip from stem (c1)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv18 = ConvModule(c2 + c1, c1, 3, 1, 1, bn_momentum)
        self.conv19 = ConvModule(c1, c1, 3, 1, 1, bn_momentum)
        
        # Final output layers (no additional upsampling needed - stem preserved resolution)
        self.conv_out = nn.Sequential(
            nn.Conv2d(c1, 3, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Motion prompt processing
        motion_input = x.view(-1, 3, 3, self.input_height, self.input_width)
        residual_maps, motion_loss = self.motion_prompt(motion_input)

        # Encoder
        features = self.backbone(x)
        # features = [stem(c1), stage1(c2), stage2(c3), stage3(c4), stage4(c5)]
        x_stem, x2, x3, x4, x_bottleneck = features

        # Decoder with skip connections
        # Upsample 1
        u1 = self.up1(x_bottleneck)
        c1 = torch.cat([u1, x4], dim=1)
        x5 = self.conv13(self.conv12(self.conv11(c1)))

        # Upsample 2
        u2 = self.up2(x5)
        c2 = torch.cat([u2, x3], dim=1)
        x6 = self.conv15(self.conv14(c2))

        # Upsample 3
        u3 = self.up3(x6)
        c3 = torch.cat([u3, x2], dim=1)
        x7 = self.conv17(self.conv16(c3))
        
        # Upsample 4
        u4 = self.up4(x7)
        c4 = torch.cat([u4, x_stem], dim=1)
        x8 = self.conv19(self.conv18(c4))

        # Final output with fusion
        x = self.conv_out[0](x8)  # Conv to 3 channels
        x = self.fusion_layer(x, residual_maps)  # Apply motion fusion
        out = self.conv_out[1](x)  # Apply sigmoid
        
        return out, motion_loss


if __name__ == "__main__":
    # Test the model
    model = TrackNetV4_CSPNeXt_BallTracking(
        input_height=288, 
        input_width=512,
        fusion_layer_type="TypeA",
        deepen_factor=1.0,
        widen_factor=0.5,
        bn_momentum=0.1
    )
    
    # Input: 3 frames concatenated (B, 9, H, W)
    x = torch.randn(2, 9, 288, 512)
    
    output, loss = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Motion loss: {loss.item()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Verify output shape
    assert output.shape == (2, 3, 288, 512), f"Expected (2, 3, 288, 512), got {output.shape}"
    print("\n✓ Model test passed!")
