"""
Hybrid Attention Knowledge Distillation Module
Combines Channel Attention and Spatial Attention for improved knowledge transfer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridAttentionLoss(nn.Module):
    """Hybrid Attention Transfer combining Channel and Spatial Attention.
    
    This module implements CBAM-style attention that preserves both
    channel-wise importance (what) and spatial importance (where).
    """
    
    def __init__(self, channels_s, channels_t, p=2, beta=1000.0, temperature=3.0):
        """
        Args:
            channels_s: List of student channel dimensions
            channels_t: List of teacher channel dimensions  
            p: Power parameter for channel attention
            beta: Temperature for spatial attention normalization
            temperature: Temperature scaling for attention maps
        """
        super().__init__()
        self.p = p
        self.beta = beta
        self.temperature = temperature
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Channel attention modules (SE-style)
        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(t_chan, max(t_chan // 16, 1), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(t_chan // 16, 1), t_chan, 1),
                nn.Sigmoid()
            ).to(device) for t_chan in channels_t
        ])
        
        # Spatial attention modules
        self.spatial_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            ).to(device) for _ in channels_t
        ])

    def channel_attention_map(self, fm, idx):
        """Calculate channel attention using SE-style attention.
        
        Args:
            fm: Feature map of shape (N, C, H, W)
            idx: Layer index for module selection
            
        Returns:
            Channel attention weights of shape (N, C, 1, 1)
        """
        if idx < len(self.channel_attention):
            return self.channel_attention[idx](fm)
        else:
            # Fallback to power-based attention
            N, C = fm.shape[:2]
            fm_pool = F.adaptive_avg_pool2d(fm, (1, 1))  # (N, C, 1, 1)
            
            # L2 norm pooling
            if self.p == 2:
                attention = torch.sqrt(torch.abs(fm_pool) + 1e-6)
            else:
                attention = torch.pow(torch.abs(fm_pool) + 1e-6, self.p)
            
            # Normalize across channels
            attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-6)
            return attention
            
    def spatial_attention_map(self, fm, idx):
        """Calculate spatial attention using max and avg pooling.
        
        Args:
            fm: Feature map of shape (N, C, H, W)
            idx: Layer index for module selection
            
        Returns:
            Spatial attention map of shape (N, 1, H, W)
        """
        # Channel pooling (max and avg)
        avg_pool = torch.mean(fm, dim=1, keepdim=True)  # (N, 1, H, W)
        max_pool, _ = torch.max(fm, dim=1, keepdim=True)  # (N, 1, H, W)
        
        # Concatenate and apply convolution
        if idx < len(self.spatial_conv):
            pool_concat = torch.cat([avg_pool, max_pool], dim=1)  # (N, 2, H, W)
            spatial_attn = self.spatial_conv[idx](pool_concat)  # (N, 1, H, W)
        else:
            # Fallback to simple spatial attention
            spatial_attn = torch.mean(torch.abs(fm), dim=1, keepdim=True)
            
            # Normalize with softmax
            N, _, H, W = spatial_attn.shape
            spatial_attn_flat = spatial_attn.view(N, -1)
            spatial_attn_flat = F.softmax(spatial_attn_flat * self.beta, dim=1)
            spatial_attn = spatial_attn_flat.view(N, 1, H, W)
            
        return spatial_attn
        
    def hybrid_attention(self, fm, idx):
        """Combine channel and spatial attention.
        
        Args:
            fm: Feature map of shape (N, C, H, W)
            idx: Layer index
            
        Returns:
            Hybrid attention map of shape (N, C, H, W)
        """
        # Get channel attention
        channel_attn = self.channel_attention_map(fm, idx)  # (N, C, 1, 1)
        
        # Get spatial attention
        spatial_attn = self.spatial_attention_map(fm, idx)  # (N, 1, H, W)
        
        # Combine: element-wise multiplication
        # First apply channel attention
        fm_channel = fm * channel_attn
        
        # Then apply spatial attention
        fm_hybrid = fm_channel * spatial_attn
        
        # Also compute combined attention map for loss calculation
        hybrid_attn = channel_attn * spatial_attn  # (N, C, H, W)
        
        # Apply temperature scaling
        hybrid_attn = hybrid_attn / self.temperature
        
        return hybrid_attn
        
    def forward(self, y_s, y_t):
        """Forward computation.
        
        Args:
            y_s (list): Student feature maps
            y_t (list): Teacher feature maps
            
        Returns:
            torch.Tensor: Hybrid attention transfer loss
        """
        assert len(y_s) == len(y_t), f"Mismatch: {len(y_s)} vs {len(y_t)}"
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # Student features are already aligned by FeatureLoss
            # No need for additional alignment here

            # Calculate hybrid attention maps
            s_attention = self.hybrid_attention(s, idx)
            t_attention = self.hybrid_attention(t, idx)
            
            # Calculate loss
            # Use both L2 and KL divergence for better gradient flow
            l2_loss = F.mse_loss(s_attention, t_attention)
            
            # Flatten and apply softmax for KL divergence
            s_flat = s_attention.view(s_attention.size(0), -1)
            t_flat = t_attention.view(t_attention.size(0), -1)
            
            s_log_softmax = F.log_softmax(s_flat, dim=1)
            t_softmax = F.softmax(t_flat, dim=1)
            
            kl_loss = F.kl_div(s_log_softmax, t_softmax, reduction='batchmean')
            
            # Combine losses with weighting
            loss = 0.7 * l2_loss + 0.3 * kl_loss
            losses.append(loss)
            
        return sum(losses) / len(losses)


def test_hybrid_attention():
    """Test the hybrid attention module."""
    # Test configuration
    batch_size = 2
    channels_s = [64, 128, 256]
    channels_t = [128, 256, 512]
    H, W = 32, 32
    
    # Create dummy features
    y_s = [torch.randn(batch_size, c, H, W) for c in channels_s]
    y_t = [torch.randn(batch_size, c, H, W) for c in channels_t]
    
    # Create hybrid attention loss
    hybrid_loss = HybridAttentionLoss(channels_s, channels_t)
    
    # Move to CUDA if available
    if torch.cuda.is_available():
        y_s = [y.cuda() for y in y_s]
        y_t = [y.cuda() for y in y_t]
        hybrid_loss = hybrid_loss.cuda()
    
    # Calculate loss
    loss = hybrid_loss(y_s, y_t)
    
    print(f"Hybrid Attention Loss: {loss.item():.4f}")
    print("Test passed!")


if __name__ == "__main__":
    test_hybrid_attention()