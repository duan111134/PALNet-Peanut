import torch
import argparse
import csv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import tifffile
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
import os
import logging
from datetime import datetime
import cv2
import random
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler
import math
from torchvision.models import resnet50
from sklearn.metrics import average_precision_score

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


# 设置随机种子以确保结果可重复
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ECA注意力机制模块 - 允许调整参数
class ECAModule(nn.Module):
    """Efficient Channel Attention module with adjustable parameters"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        # 根据通道数动态计算卷积核大小
        t = int(abs(math.log(channels, 2) + b) / gamma)
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 特征图形状[B, C, H, W]
        y = self.avg_pool(x)  # 输出形状[B, C, 1, 1]
        
        # 将通道维度转换为特征维度以进行1D卷积
        y = y.squeeze(-1).transpose(-1, -2)  # 输出形状[B, 1, C]
        y = self.conv(y)  # 输出形状[B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # 输出形状[B, C, 1, 1]
        
        # 应用sigmoid激活函数并与原始特征相乘
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# 空间注意力模块 - 比混合注意力更轻量级
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 生成通道维度上的最大值和平均值
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # 拼接最大值和平均值
        pool = torch.cat([max_pool, avg_pool], dim=1)
        
        # 应用卷积和sigmoid获取空间注意力图
        attention = self.sigmoid(self.conv(pool))
        
        # 将注意力图应用到输入特征上
        return x * attention


# 带有优化注意力机制的UNet - 简化版本
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256], dropout_rate=0.2):
        super(ImprovedUNet, self).__init__()
        
        # 方案2：只在部分编码器block后插入ECA（低层和高层）
        # 第一层（低层）插入ECA
        self.encoder1 = nn.Sequential(
            self._double_conv(in_channels, features[0]),
            ECAModule(features[0], gamma=2, b=1)
        )
        
        # 第二层不插入注意力
        self.encoder2 = self._double_conv(features[0], features[1])
        
        # 第三层不插入注意力
        self.encoder3 = self._double_conv(features[1], features[2])
        
        # 第四层（高层）插入ECA
        self.encoder4 = nn.Sequential(
            self._double_conv(features[2], features[3]),
            ECAModule(features[3], gamma=2, b=1)
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout_rate)

        # 瓶颈层，使用ECA专注于通道信息
        self.bottleneck = nn.Sequential(
            self._double_conv(features[3], features[3] * 2),
            ECAModule(features[3] * 2, gamma=2, b=1)
        )

        # 解码器路径 - 移除注意力机制以测试纯编码器注意力的效果
        self.upconv4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.decoder4 = self._double_conv(features[3] * 2, features[3])

        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = self._double_conv(features[2] * 2, features[2])

        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = self._double_conv(features[1] * 2, features[1])

        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = self._double_conv(features[0] * 2, features[0])
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 下采样路径（编码器）
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc2 = self.dropout(enc2)

        enc3 = self.encoder3(self.pool(enc2))
        enc3 = self.dropout(enc3)

        enc4 = self.encoder4(self.pool(enc3))
        enc4 = self.dropout(enc4)

        # 瓶颈
        bottleneck = self.bottleneck(self.pool(enc4))
        bottleneck = self.dropout(bottleneck)

        # 上采样路径（解码器）- 使用简单拼接而非复杂的特征融合
        dec4 = self.upconv4(bottleneck)
        if dec4.shape != enc4.shape:
            dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=True)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        if dec3.shape != enc3.shape:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=True)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        if dec2.shape != enc2.shape:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        if dec1.shape != enc1.shape:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # 直接输出最终卷积结果，不使用多尺度融合
        return self.final_conv(dec1)


# 深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 残差连接模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out


# 使用深度可分离卷积的双卷积块
def double_depthwise_conv(in_channels, out_channels):
    return nn.Sequential(
        DepthwiseSeparableConv(in_channels, out_channels),
        DepthwiseSeparableConv(out_channels, out_channels)
    )


# 多尺度融合UNet模型
class MultiScaleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256], use_residual=True, use_depthwise=False):
        super(MultiScaleUNet, self).__init__()
        
        self.use_residual = use_residual
        self.use_depthwise = use_depthwise
        
        # 编码器路径
        if use_residual:
            self.encoder1 = ResidualBlock(in_channels, features[0])
            self.encoder2 = ResidualBlock(features[0], features[1])
            self.encoder3 = ResidualBlock(features[1], features[2])
            self.encoder4 = ResidualBlock(features[2], features[3])
        elif use_depthwise:
            self.encoder1 = double_depthwise_conv(in_channels, features[0])
            self.encoder2 = double_depthwise_conv(features[0], features[1])
            self.encoder3 = double_depthwise_conv(features[1], features[2])
            self.encoder4 = double_depthwise_conv(features[2], features[3])
        else:
            self.encoder1 = self._double_conv(in_channels, features[0])
            self.encoder2 = self._double_conv(features[0], features[1])
            self.encoder3 = self._double_conv(features[1], features[2])
            self.encoder4 = self._double_conv(features[2], features[3])
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 瓶颈层
        if use_residual:
            self.bottleneck = ResidualBlock(features[3], features[3] * 2)
        elif use_depthwise:
            self.bottleneck = double_depthwise_conv(features[3], features[3] * 2)
        else:
            self.bottleneck = self._double_conv(features[3], features[3] * 2)
            
        # ECA注意力
        self.eca_bottleneck = ECAModule(features[3] * 2)
        
        # 解码器路径
        self.upconv4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        
        if use_residual:
            self.decoder4 = ResidualBlock(features[3] * 2, features[3])
            self.decoder3 = ResidualBlock(features[2] * 2, features[2])
            self.decoder2 = ResidualBlock(features[1] * 2, features[1])
            self.decoder1 = ResidualBlock(features[0] * 2, features[0])
        elif use_depthwise:
            self.decoder4 = double_depthwise_conv(features[3] * 2, features[3])
            self.decoder3 = double_depthwise_conv(features[2] * 2, features[2])
            self.decoder2 = double_depthwise_conv(features[1] * 2, features[1])
            self.decoder1 = double_depthwise_conv(features[0] * 2, features[0])
        else:
            self.decoder4 = self._double_conv(features[3] * 2, features[3])
            self.decoder3 = self._double_conv(features[2] * 2, features[2])
            self.decoder2 = self._double_conv(features[1] * 2, features[1])
            self.decoder1 = self._double_conv(features[0] * 2, features[0])
        
        # 多尺度特征融合
        self.msf_conv1 = nn.Conv2d(features[0], features[0], kernel_size=1)
        self.msf_conv2 = nn.Conv2d(features[1], features[0], kernel_size=1)
        self.msf_conv3 = nn.Conv2d(features[2], features[0], kernel_size=1)
        self.msf_conv4 = nn.Conv2d(features[3], features[0], kernel_size=1)
        
        # 最终融合卷积
        self.fusion_conv = nn.Conv2d(features[0] * 4, features[0], kernel_size=1)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码器路径
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # 瓶颈
        bottleneck = self.bottleneck(self.pool(enc4))
        bottleneck = self.eca_bottleneck(bottleneck)
        
        # 解码器路径
        dec4 = self.upconv4(bottleneck)
        if dec4.shape != enc4.shape:
            dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=True)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        if dec3.shape != enc3.shape:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=True)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        if dec2.shape != enc2.shape:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        if dec1.shape != enc1.shape:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # 多尺度特征融合
        # 将各层特征上采样到相同大小
        _, _, H, W = dec1.size()
        
        f1 = self.msf_conv1(dec1)
        
        f2 = self.msf_conv2(dec2)
        f2 = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=True)
        
        f3 = self.msf_conv3(dec3)
        f3 = F.interpolate(f3, size=(H, W), mode='bilinear', align_corners=True)
        
        f4 = self.msf_conv4(dec4)
        f4 = F.interpolate(f4, size=(H, W), mode='bilinear', align_corners=True)
        
        # 拼接多尺度特征
        fused = torch.cat([f1, f2, f3, f4], dim=1)
        
        # 融合特征
        fused = self.fusion_conv(fused)
        
        # 最终输出
        output = self.final_conv(fused)
        
        # 确保输出尺寸与目标尺寸一致 (640x640)
        if output.shape[2] != 640 or output.shape[3] != 640:
            output = F.interpolate(output, size=(640, 640), mode='bilinear', align_corners=True)
        
        return output


# 普通卷积编码器（用于消融实验 - 不使用ResNet50）
class BaselineEncoder(nn.Module):
    def __init__(self):
        super(BaselineEncoder, self).__init__()
        # 使用普通卷积替代ResNet50
        # 输出通道数设计为与ResNet50类似：[64, 256, 512, 1024, 2048]
        
        # Stage 0: 初始卷积
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # 输出: 64, H/2, W/2
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 输出: 64, H/4, W/4
        
        # Stage 1
        self.layer1 = self._make_layer(64, 256, num_blocks=2)    # 输出: 256, H/4, W/4
        # Stage 2
        self.layer2 = self._make_layer(256, 512, num_blocks=2, stride=2)   # 输出: 512, H/8, W/8
        # Stage 3
        self.layer3 = self._make_layer(512, 1024, num_blocks=2, stride=2)  # 输出: 1024, H/16, W/16
        # Stage 4
        self.layer4 = self._make_layer(1024, 2048, num_blocks=2, stride=2) # 输出: 2048, H/32, W/32
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        # 第一个block可能有stride和通道变换
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        # 其余block保持通道和尺寸不变
        for _ in range(1, num_blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 提取各层特征
        x0 = self.initial(x)           # 64, H/2, W/2
        x1 = self.maxpool(x0)          # 64, H/4, W/4
        x2 = self.layer1(x1)           # 256, H/4, W/4
        x3 = self.layer2(x2)           # 512, H/8, W/8
        x4 = self.layer3(x3)           # 1024, H/16, W/16
        x5 = self.layer4(x4)           # 2048, H/32, W/32
        
        return x0, x2, x3, x4, x5  # 返回多个stage特征


# ResNet50编码器，输出多层特征
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        resnet = resnet50(pretrained=pretrained)
        
        # 编码器各层
        self.initial = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu
        )  # 输出: 64, H/2, W/2
        self.maxpool = resnet.maxpool  # 输出: 64, H/4, W/4
        self.layer1 = resnet.layer1    # 输出: 256, H/4, W/4
        self.layer2 = resnet.layer2    # 输出: 512, H/8, W/8
        self.layer3 = resnet.layer3    # 输出: 1024, H/16, W/16
        self.layer4 = resnet.layer4    # 输出: 2048, H/32, W/32

    def forward(self, x):
        # 提取各层特征
        x0 = self.initial(x)           # 64, H/2, W/2
        x1 = self.maxpool(x0)          # 64, H/4, W/4
        x2 = self.layer1(x1)           # 256, H/4, W/4
        x3 = self.layer2(x2)           # 512, H/8, W/8
        x4 = self.layer3(x3)           # 1024, H/16, W/16
        x5 = self.layer4(x4)           # 2048, H/32, W/32
        
        return x0, x2, x3, x4, x5  # 返回多个stage特征


# 通用可配置模型（支持所有消融组合）
class ConfigurableUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, use_resnet=True, use_fpn=True, use_eca=True, dropout_rate=0.2):
        super(ConfigurableUNet, self).__init__()
        
        self.use_fpn = use_fpn
        self.use_eca = use_eca
        
        # 编码器选择：ResNet50 或 Baseline
        if use_resnet:
            self.encoder = ResNet50Encoder(pretrained=True)
        else:
            self.encoder = BaselineEncoder()
        
        self.features = [64, 256, 512, 1024, 2048]
        self.fpn_channel = 256
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # ECA注意力（可选）
        if use_eca:
            self.eca_bottleneck = ECAModule(self.features[4])
        else:
            self.eca_bottleneck = nn.Identity()  # 跳过ECA
        
        if use_fpn:
            # 使用标准FPN结构
            self.lateral_conv5 = nn.Conv2d(self.features[4], self.fpn_channel, kernel_size=1)
            self.lateral_conv4 = nn.Conv2d(self.features[3], self.fpn_channel, kernel_size=1)
            self.lateral_conv3 = nn.Conv2d(self.features[2], self.fpn_channel, kernel_size=1)
            self.lateral_conv2 = nn.Conv2d(self.features[1], self.fpn_channel, kernel_size=1)
            self.lateral_conv1 = nn.Conv2d(self.features[0], self.fpn_channel, kernel_size=1)
            self.fpn_conv5 = nn.Conv2d(self.fpn_channel, self.fpn_channel, kernel_size=3, padding=1)
            self.fpn_conv4 = nn.Conv2d(self.fpn_channel, self.fpn_channel, kernel_size=3, padding=1)
            self.fpn_conv3 = nn.Conv2d(self.fpn_channel, self.fpn_channel, kernel_size=3, padding=1)
            self.fpn_conv2 = nn.Conv2d(self.fpn_channel, self.fpn_channel, kernel_size=3, padding=1)
            self.fpn_conv1 = nn.Conv2d(self.fpn_channel, self.fpn_channel, kernel_size=3, padding=1)
            self.decoder_conv4 = self._double_conv(self.fpn_channel, self.fpn_channel)
            self.decoder_conv3 = self._double_conv(self.fpn_channel, self.fpn_channel)
            self.decoder_conv2 = self._double_conv(self.fpn_channel, self.fpn_channel)
            self.decoder_conv1 = self._double_conv(self.fpn_channel, self.fpn_channel)
            self.final_conv = nn.Conv2d(self.fpn_channel, out_channels, kernel_size=1)
        else:
            # 不使用FPN，使用简单的UNet解码器
            self.upconv4 = nn.ConvTranspose2d(self.features[4], self.features[3], kernel_size=2, stride=2)
            self.decoder4 = self._double_conv(self.features[3] * 2, self.features[3])
            
            self.upconv3 = nn.ConvTranspose2d(self.features[3], self.features[2], kernel_size=2, stride=2)
            self.decoder3 = self._double_conv(self.features[2] * 2, self.features[2])
            
            self.upconv2 = nn.ConvTranspose2d(self.features[2], self.features[1], kernel_size=2, stride=2)
            self.decoder2 = self._double_conv(self.features[1] * 2, self.features[1])
            
            self.upconv1 = nn.ConvTranspose2d(self.features[1], self.features[0], kernel_size=2, stride=2)
            self.decoder1 = self._double_conv(self.features[0] * 2, self.features[0])
            
            self.final_conv = nn.Conv2d(self.features[0], out_channels, kernel_size=1)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=True) + y
    
    def forward(self, x):
        enc1, enc2, enc3, enc4, enc5 = self.encoder(x)
        enc5 = self.eca_bottleneck(enc5)
        enc5 = self.dropout(enc5)
        
        if self.use_fpn:
            # FPN路径
            p5 = self.lateral_conv5(enc5)
            p4 = self._upsample_add(p5, self.lateral_conv4(enc4))
            p3 = self._upsample_add(p4, self.lateral_conv3(enc3))
            p2 = self._upsample_add(p3, self.lateral_conv2(enc2))
            p1 = self._upsample_add(p2, self.lateral_conv1(enc1))

            p5 = self.fpn_conv5(p5)
            p4 = self.fpn_conv4(p4)
            p3 = self.fpn_conv3(p3)
            p2 = self.fpn_conv2(p2)
            p1 = self.fpn_conv1(p1)

            d4 = self.decoder_conv4(p4)
            d3 = self.decoder_conv3(p3)
            d2 = self.decoder_conv2(p2)
            d1 = self.decoder_conv1(p1)
            out = d1
        else:
            # 标准UNet解码器路径
            dec4 = self.upconv4(enc5)
            if dec4.shape != enc4.shape:
                dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=True)
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            
            dec3 = self.upconv3(dec4)
            if dec3.shape != enc3.shape:
                dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=True)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            
            dec2 = self.upconv2(dec3)
            if dec2.shape != enc2.shape:
                dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=True)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            
            dec1 = self.upconv1(dec2)
            if dec1.shape != enc1.shape:
                dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=True)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1)
            out = dec1
        
        out = self.final_conv(out)
        if out.shape[2] != 640 or out.shape[3] != 640:
            out = F.interpolate(out, size=(640, 640), mode='bilinear', align_corners=True)
        return out


# ResNet50+FPN分割网络
class ResNet50_FPN_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, pretrained=True, dropout_rate=0.2):
        super(ResNet50_FPN_UNet, self).__init__()
        self.encoder = ResNet50Encoder(pretrained=pretrained)
        self.features = [64, 256, 512, 1024, 2048]
        self.fpn_channel = 256
        self.dropout = nn.Dropout2d(dropout_rate)
        # FPN横向连接
        self.lateral_conv5 = nn.Conv2d(self.features[4], self.fpn_channel, kernel_size=1)
        self.lateral_conv4 = nn.Conv2d(self.features[3], self.fpn_channel, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(self.features[2], self.fpn_channel, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(self.features[1], self.fpn_channel, kernel_size=1)
        self.lateral_conv1 = nn.Conv2d(self.features[0], self.fpn_channel, kernel_size=1)
        # FPN自顶向下路径
        self.fpn_conv5 = nn.Conv2d(self.fpn_channel, self.fpn_channel, kernel_size=3, padding=1)
        self.fpn_conv4 = nn.Conv2d(self.fpn_channel, self.fpn_channel, kernel_size=3, padding=1)
        self.fpn_conv3 = nn.Conv2d(self.fpn_channel, self.fpn_channel, kernel_size=3, padding=1)
        self.fpn_conv2 = nn.Conv2d(self.fpn_channel, self.fpn_channel, kernel_size=3, padding=1)
        self.fpn_conv1 = nn.Conv2d(self.fpn_channel, self.fpn_channel, kernel_size=3, padding=1)
        # 解码器路径
        self.decoder_conv4 = self._double_conv(self.fpn_channel, self.fpn_channel)
        self.decoder_conv3 = self._double_conv(self.fpn_channel, self.fpn_channel)
        self.decoder_conv2 = self._double_conv(self.fpn_channel, self.fpn_channel)
        self.decoder_conv1 = self._double_conv(self.fpn_channel, self.fpn_channel)
        # ECA注意力
        self.eca_bottleneck = ECAModule(self.features[4])
        # 最终输出层
        self.final_conv = nn.Conv2d(self.fpn_channel, out_channels, kernel_size=1)
        # 空间注意力（可选）
        # self.spatial_attention = SpatialAttention(kernel_size=7)
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=True) + y
    def forward(self, x):
        enc1, enc2, enc3, enc4, enc5 = self.encoder(x)
        enc5 = self.eca_bottleneck(enc5)
        enc5 = self.dropout(enc5)
        # FPN横向连接
        p5 = self.lateral_conv5(enc5)
        p4 = self._upsample_add(p5, self.lateral_conv4(enc4))
        p3 = self._upsample_add(p4, self.lateral_conv3(enc3))
        p2 = self._upsample_add(p3, self.lateral_conv2(enc2))
        p1 = self._upsample_add(p2, self.lateral_conv1(enc1))
        # FPN自顶向下路径
        p5 = self.fpn_conv5(p5)
        p4 = self.fpn_conv4(p4)
        p3 = self.fpn_conv3(p3)
        p2 = self.fpn_conv2(p2)
        p1 = self.fpn_conv1(p1)
        # 解码器路径
        d4 = self.decoder_conv4(p4)
        d3 = self.decoder_conv3(p3)
        d2 = self.decoder_conv2(p2)
        d1 = self.decoder_conv1(p1)
        out = d1
        # 可选：空间注意力
        # out = self.spatial_attention(out)
        out = self.final_conv(out)
        if out.shape[2] != 640 or out.shape[3] != 640:
            out = F.interpolate(out, size=(640, 640), mode='bilinear', align_corners=True)
        return out


# 读取已确定的论文划分；公开版不重新随机划分数据。
def load_split_csv(split_csv):
    split_csv = Path(split_csv)
    with split_csv.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))

    required = {'image', 'annotation', 'split'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Split CSV must contain columns: {sorted(required)}")

    allowed = {'train', 'validation', 'test'}
    records = []
    seen_images = set()
    seen_annotations = set()
    for row in rows:
        image = row['image'].strip()
        annotation = row['annotation'].strip()
        split = row['split'].strip().lower()
        if not image or not annotation or split not in allowed:
            raise ValueError(f"Invalid split row: {row}")
        if image in seen_images or annotation in seen_annotations:
            raise ValueError(f"Duplicate image or annotation in split CSV: {row}")
        if Path(image).stem != Path(annotation).stem:
            raise ValueError(f"Image/annotation stems do not match: {row}")
        seen_images.add(image)
        seen_annotations.add(annotation)
        records.append({'image': image, 'annotation': annotation, 'split': split})

    counts = {name: sum(r['split'] == name for r in records) for name in sorted(allowed)}
    total_count = sum(counts.values())
    if total_count != len(records):
        raise ValueError(
            f"Split counts do not sum to the number of records: {counts}; total={total_count}"
        )
    return records


# 数据加载器类，支持数据增强
class PlantDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, augment=True, file_names=None):  # 启用数据增强
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.transform = transform
        self.augment = augment  # 使用传入的augment参数值

        # 文件名来自已保存的论文划分；未提供时保留原始目录扫描行为。
        self.image_files = (
            [self.image_dir / name for name in file_names]
            if file_names is not None else list(self.image_dir.glob('*.tif'))
        )
        self.valid_samples = []

        # 验证文件
        for img_path in self.image_files:
            base_name = img_path.stem
            annotation_path = self.annotation_dir / f"{base_name}.json"

            if annotation_path.exists():
                try:
                    # 读取并验证标注文件
                    with open(annotation_path, 'r', encoding='utf-8') as f:
                        annotations = json.load(f)

                    # 检查标注格式
                    if isinstance(annotations, list):  # 如果标注是点列表
                        points = annotations
                    elif isinstance(annotations, dict):  # 如果标注是字典
                        if 'points' in annotations:
                            points = annotations['points']
                        elif 'annotations' in annotations:
                            points = annotations['annotations']
                        elif 'shapes' in annotations:  # 处理labelme格式
                            points = []
                            for shape in annotations['shapes']:
                                if shape['shape_type'] == 'point':
                                    points.append(shape['points'][0])
                        else:
                            logging.warning(f"标注文件 {annotation_path} 格式不正确，跳过")
                            continue
                    else:
                        logging.warning(f"标注文件 {annotation_path} 格式不正确，跳过")
                        continue

                    # 验证点的格式
                    if not isinstance(points, list):
                        logging.warning(f"标注文件 {annotation_path} 中的点格式不正确，跳过")
                        continue

                    # 验证每个点的格式
                    valid_points = []
                    for point in points:
                        if isinstance(point, (list, tuple)) and len(point) == 2:
                            try:
                                x, y = float(point[0]), float(point[1])
                                valid_points.append([x, y])
                            except (ValueError, TypeError):
                                logging.warning(f"点坐标格式不正确 {point}")
                                continue

                    if not valid_points:
                        logging.warning(f"标注文件 {annotation_path} 中没有有效的点，跳过")
                        continue

                    self.valid_samples.append({
                        'image': img_path,
                        'annotation': annotation_path,
                        'points': valid_points
                    })

                except json.JSONDecodeError as e:
                    logging.error(f"JSON解析错误 {annotation_path} {str(e)}")
                    continue
                except Exception as e:
                    logging.error(f"处理标注文件 {annotation_path} 时出错 {str(e)}")
                    continue

        logging.info(f"找到 {len(self.valid_samples)} 个有效样本")

    def __len__(self):
        return len(self.valid_samples)

    def apply_augmentation(self, image, mask):
        # 随机水平翻转
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # 随机垂直翻转
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        # 随机旋转
        angle = random.choice([0, 90, 180, 270])
        if angle > 0:
            image = np.rot90(image, k=angle // 90).copy()
            mask = np.rot90(mask, k=angle // 90).copy()

        # 随机亮度和对比度调整
        if random.random() > 0.5:
            alpha = 0.8 + random.random() * 0.4
            beta = -10 + random.random() * 20
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

        # 随机高斯噪声
        if random.random() > 0.7:
            noise = np.random.normal(0, 5, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        return image, mask

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]

        try:
            # 读取图像
            image = tifffile.imread(str(sample['image']))
            
            # 确保图像是3通道的
            if len(image.shape) == 2:  # 如果是单通道图像
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[-1] == 4:  # 如果是RGBA图像
                image = image[..., :3]  # 只取RGB通道
            
            # 创建分割掩码
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            for point in sample['points']:
                x, y = point
                # 使用高斯核替代圆形掩码
                sigma = 10  # 训练/评估统一的高斯sigma
                # 覆盖到约3σ，确保kernel_size为奇数
                kernel_size = int(sigma * 6 + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                # 创建网格坐标
                x_grid, y_grid = np.meshgrid(
                    np.arange(max(0, int(x) - kernel_size // 2), min(mask.shape[1], int(x) + kernel_size // 2 + 1)),
                    np.arange(max(0, int(y) - kernel_size // 2), min(mask.shape[0], int(y) + kernel_size // 2 + 1))
                )
                
                # 计算高斯值
                gaussian = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
                
                # 更新掩码
                mask[y_grid, x_grid] = np.maximum(mask[y_grid, x_grid], gaussian)

            # 应用数据增强
            if self.augment and random.random() > 0.5:
                image, mask = self.apply_augmentation(image, mask)

            # 确保所有图像尺寸相同 (调整为固定大小)
            target_size = (640, 640)  # 选择一个合适的尺寸
            if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
                image = cv2.resize(image, target_size)
                mask = cv2.resize(mask, target_size)

            # 转换为tensor
            if len(image.shape) == 3:  # 确保是3通道
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            else:  # 如果是单通道，扩展为3通道
                image = torch.from_numpy(np.stack([image, image, image], axis=0)).float() / 255.0
                
            mask = torch.from_numpy(mask).float().unsqueeze(0)

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)

            return image, mask

        except Exception as e:
            logging.error(f"处理样本 {sample['image']} 时出错 {str(e)}")
            # 返回一个与其他样本大小相同的空张量
            return torch.zeros((3, 640, 640)), torch.zeros((1, 640, 640))


# 组合损失函数
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss()

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(torch.sigmoid(inputs), targets)
        focal = self.focal_loss(inputs, targets)
        return self.alpha * bce + self.beta * dice + (1 - self.alpha - self.beta) * focal


# Dice损失
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


# Focal损失
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path, scheduler=None,
                patience=None):
    # 初始化训练参数
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    current_batch_size = train_loader.batch_size

    # 初始化混合精度训练
    scaler = GradScaler()

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0

        # 训练阶段
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]") as t:
            for images, densities in t:
                try:
                    images = images.to(device, non_blocking=True)
                    densities = densities.to(device, non_blocking=True)

                    # 混合精度训练
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, densities)

                    # 反向传播
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()

                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item()
                    batch_count += 1
                    t.set_postfix(loss=loss.item())

                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        logging.error(f"GPU内存不足 {str(e)}")
                        torch.cuda.empty_cache()
                        # 如果OOM，尝试清理缓存并继续
                        logging.info("清理GPU缓存并继续训练...")
                        continue
                    else:
                        logging.error(f"训练时发生错误 {str(e)}")
                        raise e

        # 计算平均训练损失
        if batch_count > 0:  # 确保不会除以零
            avg_train_loss = train_loss / batch_count
        else:
            avg_train_loss = float('inf')
            
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]") as t:
                for images, densities in t:
                    try:
                        images = images.to(device, non_blocking=True)
                        densities = densities.to(device, non_blocking=True)

                        with autocast():
                            outputs = model(images)
                            loss = criterion(outputs, densities)

                        val_loss += loss.item()
                        val_batch_count += 1
                        t.set_postfix(loss=loss.item())

                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            logging.error(f"GPU内存不足 {str(e)}")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            logging.error(f"验证时发生错误 {str(e)}")
                            raise e

        # 计算平均验证损失
        if val_batch_count > 0:  # 确保不会除以零
            avg_val_loss = val_loss / val_batch_count
        else:
            avg_val_loss = float('inf')
            
        val_losses.append(avg_val_loss)

        # 学习率调整
        if scheduler:
            if isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
            else:
                scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path.mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path / 'best_unet.pth')
            logging.info(f"Epoch {epoch + 1} 保存最佳模型，验证损失 = {avg_val_loss:.4f}")

        # 保存检查点
        if (epoch + 1) % 5 == 0:  # 降低保存频率，减少磁盘IO
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler': scaler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, save_path / f'checkpoint_epoch_{epoch + 1}.pth')

        logging.info(f"Epoch {epoch + 1} 训练损失 = {avg_train_loss:.4f}, 验证损失 = {avg_val_loss:.4f}")

        # 主动清理GPU内存
        torch.cuda.empty_cache()

        # 早停机制
        if patience and epoch >= patience and avg_val_loss >= min(val_losses[-patience:]):
            logging.info(f"早停机制触发，验证损失在{patience}个epoch内没有改善，停止训练")
            break

    # 保存训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path / 'loss_curve.png')

    return train_losses, val_losses


def count_seedlings_from_mask(mask, threshold=0.3, min_area=15):
    """
    使用分水岭算法从分割掩码中计数植物数量
    """
    # 1. 掩码二值化
    binary_mask = (mask > threshold).astype(np.uint8) * 255

    # 2. 形态学操作去噪
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. 膨胀得到确定的背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 4. 距离变换得到确定的前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 5. 得到未知区域
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. 标记
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # 保证背景为1
    markers[unknown == 255] = 0

    # 7. 分水岭算法
    # 需要三通道图像
    mask_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mask_color, markers)

    # 8. 统计分割目标
    valid_seedlings = []
    for label in range(2, np.max(markers) + 1):  # 跳过背景和边界
        component = np.uint8(markers == label)
        area = cv2.countNonZero(component)
        if area >= min_area:
            # 计算中心点
            M = cv2.moments(component)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                valid_seedlings.append((cx, cy))

    # 9. 可视化
    result_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    for i, (x, y) in enumerate(valid_seedlings):
        cv2.circle(result_image, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(result_image, str(i + 1), (x + 5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return len(valid_seedlings), result_image, valid_seedlings


def evaluate_and_visualize(
    image_dir,
    annotation_dir,
    save_dir,
    model,
    device,
    dataset_subset=None,
    dataset_indices=None,
    sample_files=None,
    distance_threshold: float = 40.0,
    sweep_thresholds=None,
):
    
    #评估模型性能并可视化结果

    #参数
    #image_dir 图像目录
    #annotation_dir 标注目录
    #save_dir 结果保存目录
    #model 要评估的模型
    #device 计算设备
    #dataset_subset 数据集子集(训练集、验证集或测试集)
    #dataset_indices 数据集索引列表
    #distance_threshold 基准匹配阈值 r（像素）
    #sweep_thresholds 进行敏感性分析的阈值列表（含基准r）
    
    logging.info("开始评估模型性能和可视化结果...")
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # 存储所有图像的评估结果
    all_results = []
    # 用于计算AP的数据
    all_predictions = []  # 所有像素的预测概率（采样）
    all_targets = []      # 所有像素的真实标签（采样）
    # 用于速度统计
    inference_times = []  # 仅模型前向时间
    # MAE统计
    pred_counts = []
    true_counts = []

    # 处理阈值列表，确保基准值在列表中且有序去重
    sweep_thresholds = sweep_thresholds or []
    sweep_thresholds = [distance_threshold] + list(sweep_thresholds)
    sweep_thresholds = list(dict.fromkeys(sweep_thresholds))

    # 聚合敏感性统计
    sweep_stats = {
        th: {
            'precision': [],
            'recall': [],
            'f1': [],
            'matched': [],
            'pred': [],
            'true': [],
        }
        for th in sweep_thresholds
    }

    # 如果提供了文件名列表，就严格按已保存的 split 评估。
    if sample_files is not None:
        image_paths = [Path(image_dir) / name for name in sample_files]
        logging.info(f"评估已保存划分中的 {len(image_paths)} 个图像...")
    # 如果提供了数据集子集，就只评估子集中的图像
    elif dataset_subset is not None:
        # 获取子集中的文件路径
        subset_indices = dataset_subset.indices
        full_dataset = dataset_subset.dataset
        if hasattr(full_dataset, 'dataset'):  # 处理可能嵌套的Dataset
            full_dataset = full_dataset.dataset

        image_paths = [full_dataset.valid_samples[i]['image'] for i in subset_indices]
        logging.info(f"评估子集中的 {len(image_paths)} 个图像...")
    # 如果提供了索引列表，就使用索引列表
    elif dataset_indices is not None:
        # 创建一个临时数据集来获取图像路径
        temp_dataset = PlantDataset(image_dir, annotation_dir, augment=False)
        image_paths = [temp_dataset.valid_samples[i]['image'] for i in dataset_indices]
        logging.info(f"评估索引列表中的 {len(image_paths)} 个图像...")
    else:
        # 否则评估所有图像
        image_paths = list(Path(image_dir).glob('*.tif'))
        logging.info(f"评估所有 {len(image_paths)} 个图像...")

    for img_path in tqdm(image_paths, desc="评估进度"):
        try:
            # 读取图像
            image = tifffile.imread(str(img_path))
            
            # 确保图像是3通道的
            if len(image.shape) == 2:  # 如果是单通道图像
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[-1] == 4:  # 如果是RGBA图像
                image = image[..., :3]  # 只取RGB通道
            
            # 读取真实标注点
            base_name = Path(img_path).stem
            annotation_path = Path(annotation_dir) / f"{base_name}.json"

            true_points = []
            if annotation_path.exists():
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)

                if isinstance(annotations, list):
                    true_points = annotations
                elif isinstance(annotations, dict):
                    if 'points' in annotations:
                        true_points = annotations['points']
                    elif 'annotations' in annotations:
                        true_points = annotations['annotations']
                    elif 'shapes' in annotations:
                        true_points = []
                        for shape in annotations['shapes']:
                            if shape['shape_type'] == 'point':
                                true_points.append(shape['points'][0])

            # 预处理图像
            original_size = image.shape[:2]
            target_size = (640, 640)
            
            # 调整图像大小
            image_resized = cv2.resize(image, target_size)
            
            # 转换为tensor
            if len(image_resized.shape) == 3:  # 确保是3通道
                image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
            else:  # 如果是单通道，扩展为3通道
                image_tensor = torch.from_numpy(np.stack([image_resized, image_resized, image_resized], axis=0)).float() / 255.0
                
            image_tensor = image_tensor.unsqueeze(0).to(device)

            # 获取分割结果并测量纯前向时间
            with torch.no_grad():
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                logits = model(image_tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                infer_time = time.time() - start
                inference_times.append(infer_time)
                
                mask = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # 将掩码调整回原始大小
            mask_original = cv2.resize(mask, (original_size[1], original_size[0]))

            # 从分割掩码中提取点
            count, result_image, pred_points = count_seedlings_from_mask(mask, threshold=0.3, min_area=10)

            # 计算误差
            true_count = len(true_points)
            pred_count = len(pred_points)
            count_error = pred_count - true_count
            true_counts.append(true_count)
            pred_counts.append(pred_count)

            # 将标注点坐标转为浮点数列表
            true_points_float = []
            for point in true_points:
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    true_points_float.append((float(point[0]), float(point[1])))

            # 一对一匹配：使用贪心策略按距离从小到大进行配对
            # 思路：
            # 1) 计算所有预测点与真实点的两两欧氏距离
            # 2) 将(距离, 预测索引, 真实索引)按距离升序排序
            # 3) 依次尝试配对，若当前距离<=阈值且双方未匹配，则建立配对
            # 说明：该实现避免引入额外依赖（不使用scipy），结果接近最优且满足“一对一”约束

            # 将预测点坐标还原到原始图像尺寸，先缓存下来以避免重复计算
            pred_points_orig = []
            for px, py in pred_points:
                px_orig = int(px * original_size[1] / target_size[1])
                py_orig = int(py * original_size[0] / target_size[0])
                pred_points_orig.append((px_orig, py_orig))

            # 构建所有两两距离并排序
            all_pairs = []  # 元素: (dist, pi, ti)
            if len(pred_points_orig) > 0 and len(true_points_float) > 0:
                for pi, (pxo, pyo) in enumerate(pred_points_orig):
                    for ti, (tx, ty) in enumerate(true_points_float):
                        d = float(np.hypot(pxo - tx, pyo - ty))
                        all_pairs.append((d, pi, ti))
                all_pairs.sort(key=lambda x: x[0])

            def greedy_match_count(threshold_value: float) -> int:
                """基于贪心一对一匹配，返回在给定阈值下的匹配数量。"""
                if not all_pairs:
                    return 0
                matched_pred = set()
                matched_true = set()
                count = 0
                for d, pi, ti in all_pairs:
                    if d > threshold_value:
                        break  # 后续更大，无需再看
                    if (pi not in matched_pred) and (ti not in matched_true):
                        matched_pred.add(pi)
                        matched_true.add(ti)
                        count += 1
                        if count == min(len(pred_points_orig), len(true_points_float)):
                            break
                return count

            def compute_metrics(threshold_value: float):
                matched = greedy_match_count(threshold_value)
                precision_local = matched / pred_count if pred_count > 0 else 0.0
                recall_local = matched / true_count if true_count > 0 else 0.0
                f1_local = (
                    2 * precision_local * recall_local / (precision_local + recall_local)
                    if (precision_local + recall_local) > 0
                    else 0.0
                )
                return matched, precision_local, recall_local, f1_local

            # 基准阈值指标
            matched_points, precision, recall, f1 = compute_metrics(distance_threshold)

            # 敏感性阈值记录
            for th in sweep_thresholds:
                m, p, r_, f_ = compute_metrics(th)
                sweep_stats[th]['precision'].append(p)
                sweep_stats[th]['recall'].append(r_)
                sweep_stats[th]['f1'].append(f_)
                sweep_stats[th]['matched'].append(m)
                sweep_stats[th]['pred'].append(pred_count)
                sweep_stats[th]['true'].append(true_count)

            # 构建像素级标签用于AP（与训练标签口径一致，使用高斯核）
            true_mask = np.zeros((original_size[0], original_size[1]), dtype=np.float32)
            sigma_eval = 10
            kernel_size_eval = int(sigma_eval * 6 + 1)
            if kernel_size_eval % 2 == 0:
                kernel_size_eval += 1
            for point in true_points:
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    x, y = float(point[0]), float(point[1])
                    x_grid, y_grid = np.meshgrid(
                        np.arange(max(0, int(x) - kernel_size_eval // 2), min(true_mask.shape[1], int(x) + kernel_size_eval // 2 + 1)),
                        np.arange(max(0, int(y) - kernel_size_eval // 2), min(true_mask.shape[0], int(y) + kernel_size_eval // 2 + 1))
                    )
                    gaussian = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma_eval ** 2))
                    true_mask[y_grid, x_grid] = np.maximum(true_mask[y_grid, x_grid], gaussian)
            # 采样以节省内存
            sample_indices = np.random.choice(true_mask.size, min(5000, true_mask.size), replace=False)
            all_predictions.extend(mask_original.flatten()[sample_indices])
            all_targets.extend(true_mask.flatten()[sample_indices])

            # 记录结果
            img_result = {
                'image': Path(img_path).name,
                'true_count': true_count,
                'pred_count': pred_count,
                'count_error': count_error,
                'matched_points': matched_points,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            all_results.append(img_result)

            # ===== 创建纯净预测图像（无文字标注）=====
            clean_pred_image = image.copy()
            # 只绘制预测点（红色实心圆）
            for px, py in pred_points:
                cv2.circle(clean_pred_image, (px, py), 6, (255, 0, 0), -1)  # 红色预测点，稍小一点
            
            # 保存纯净预测图像
            clean_output_dir = save_dir / 'clean_predictions'
            clean_output_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(clean_output_dir / f"{Path(img_path).stem}_pred.png"), 
                       cv2.cvtColor(clean_pred_image, cv2.COLOR_RGB2BGR))

            # ===== 创建带标注的可视化结果 =====
            visual_image = image.copy()

            # 在原图上绘制预测点(红色) - 只绘制圆点，不添加文字标注
            for px, py in pred_points:
                cv2.circle(visual_image, (px, py), 8, (255, 0, 0), -1)  # 红色预测点

            # 保存可视化结果
            plt.figure(figsize=(18, 9))
            # 显示原图
            plt.subplot(121)
            plt.imshow(image)
            plt.title('原始图像')
            plt.axis('off')

            # 显示分析结果
            plt.subplot(122)
            plt.imshow(visual_image)
            plt.title(f"预测结果 (检测到 {pred_count} 个幼苗)")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(save_dir / f"{Path(img_path).stem}_analysis.png", bbox_inches='tight', dpi=300)
            plt.close()

            # 保存点和误差数据
            with open(save_dir / f"{Path(img_path).stem}_analysis.json", 'w') as f:
                json.dump(img_result, f, indent=4)

        except Exception as e:
            logging.error(f"处理图像 {img_path} 时出错 {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            continue

    if not all_results:
        logging.warning("没有成功评估任何图像!")
        empty_summary = {
            'total_images': 0,
            'avg_precision': 0,
            'avg_recall': 0,
            'avg_f1': 0,
            'true_total': 0,
            'pred_total': 0,
            'abs_error_total': 0
        }
        return all_results, empty_summary

    # 保存总体评估结果
    with open(save_dir / "evaluation_results.json", 'w') as f:
        json.dump(all_results, f, indent=4)

    # 计算并保存平均评估指标
    avg_precision = sum(r['precision'] for r in all_results) / len(all_results)
    avg_recall = sum(r['recall'] for r in all_results) / len(all_results)
    avg_f1 = sum(r['f1_score'] for r in all_results) / len(all_results)

    # AP
    try:
        if len(all_targets) > 0 and len(set(all_targets)) > 1:
            ap = average_precision_score(all_targets, all_predictions)
        else:
            ap = 0.0
    except Exception:
        ap = 0.0

    # 模型前向速度（ms与FPS）
    if len(inference_times) > 0:
        avg_infer_time = np.mean(inference_times)
        fps = 1.0 / avg_infer_time if avg_infer_time > 0 else 0.0
        prediction_speed_ms = avg_infer_time * 1000
    else:
        fps = 0.0
        prediction_speed_ms = 0.0

    # MAE
    mae = float(np.mean(np.abs(np.array(pred_counts) - np.array(true_counts)))) if len(true_counts) > 0 else 0.0

    summary = {
        'total_images': len(all_results),
        'AP': ap,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'distance_threshold': distance_threshold,
        'avg_f1': avg_f1,
        'MAE': mae,
        'FPS': fps,
        'prediction_speed_per_image_ms': prediction_speed_ms,
        'true_total': sum(r['true_count'] for r in all_results),
        'pred_total': sum(r['pred_count'] for r in all_results),
        'abs_error_total': sum(abs(r['count_error']) for r in all_results)
    }

    # 汇总敏感性分析结果（按阈值排序）
    sweep_summary = []
    for th in sorted(sweep_thresholds):
        stats = sweep_stats[th]
        n = len(stats['precision'])
        if n == 0:
            continue
        sweep_summary.append({
            'r': th,
            'avg_precision': sum(stats['precision']) / n,
            'avg_recall': sum(stats['recall']) / n,
            'avg_f1': sum(stats['f1']) / n,
        })

    summary['r_sweep'] = sweep_summary

    with open(save_dir / "evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)

    logging.info(f"评估完成! 平均准确率 {avg_precision:.4f}, 平均召回率 {avg_recall:.4f}, 平均F1分数 {avg_f1:.4f}")

    return all_results, summary


# 1. MixUp数据增强
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

# 2. 自适应数据增强
def adaptive_augment(image, mask, difficulty_score):
    # 根据模型表现动态调整增强强度
    aug_intensity = min(1.0, difficulty_score)
    # 这里先返回原始图像与掩码，避免未定义变量
    return image, mask


def parse_args():
    parser = argparse.ArgumentParser(description='PALNet point-supervised seedling localization')
    parser.add_argument('--image-dir', type=Path, default=Path('data/images'))
    parser.add_argument('--annotation-dir', type=Path, default=Path('data/annotations'))
    parser.add_argument('--split-csv', type=Path, default=Path(__file__).with_name('peanut2024_split.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    return parser.parse_args()


def main(args=None):
    # 设置随机种子
    set_seed(42)

    try:
        args = parse_args() if args is None else args
        image_dir = Path(args.image_dir)
        annotation_dir = Path(args.annotation_dir)
        split_csv = Path(args.split_csv)
        save_dir = Path(args.output_dir)

        split_records = load_split_csv(split_csv)
        split_files = {
            split: [r['image'] for r in split_records if r['split'] == split]
            for split in ('train', 'validation', 'test')
        }
        for record in split_records:
            image_name = record['image']
            annotation_name = record['annotation']
            image_path = image_dir / image_name
            annotation_path = annotation_dir / annotation_name
            if not image_path.is_file() or not annotation_path.is_file():
                raise FileNotFoundError(f"Missing split sample: {image_path} / {annotation_path}")

        # 创建输出目录
        save_dir.mkdir(parents=True, exist_ok=True)

        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"使用设备 {device}")

        # 检查CUDA信息
        if torch.cuda.is_available():
            logging.info(f"CUDA设备 {torch.cuda.get_device_name(0)}")
            logging.info(f"可用CUDA设备数量 {torch.cuda.device_count()}")
            logging.info(f"CUDA版本 {torch.version.cuda}")

        # 使用已经恢复的论文划分，不重新随机生成 split。
        train_dataset = PlantDataset(
            image_dir, annotation_dir, augment=True, file_names=split_files['train']
        )
        val_dataset = PlantDataset(
            image_dir, annotation_dir, augment=False, file_names=split_files['validation']
        )
        test_dataset = PlantDataset(
            image_dir, annotation_dir, augment=False, file_names=split_files['test']
        )

        train_size = len(train_dataset)
        val_size = len(val_dataset)
        test_size = len(test_dataset)
        total_size = train_size + val_size + test_size

        split_counts = {
            'train': sum(record['split'] == 'train' for record in split_records),
            'validation': sum(record['split'] == 'validation' for record in split_records),
            'test': sum(record['split'] == 'test' for record in split_records),
        }
        expected_sizes = (
            split_counts['train'],
            split_counts['validation'],
            split_counts['test'],
        )
        if (train_size, val_size, test_size) != expected_sizes:
            raise ValueError(
                f"PlantDataset counts do not match split CSV: "
                f"{(train_size, val_size, test_size)} != {expected_sizes}"
            )
        if total_size != sum(split_counts.values()):
            raise ValueError(
                f"Dataset total does not match split CSV: {total_size} != {sum(split_counts.values())}"
            )

        logging.info(f"数据集总大小 {total_size}")
        logging.info(f"训练集大小 {train_size}")
        logging.info(f"验证集大小 {val_size}")
        logging.info(f"测试集大小 {test_size}")

        # 使用SubsetRandomSampler创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            sampler=SubsetRandomSampler(list(range(train_size))),
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            sampler=SubsetRandomSampler(list(range(val_size))),
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=4,
            sampler=SubsetRandomSampler(list(range(test_size))),
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )

        # ============ 消融实验：模型配置 ============
        # 配置三个开关控制不同组合
        # 可以通过环境变量覆盖（用于自动化脚本）
        use_resnet = os.environ.get('ABLATION_USE_RESNET', 'True') == 'True'
        use_fpn = os.environ.get('ABLATION_USE_FPN', 'True') == 'True'
        use_eca = os.environ.get('ABLATION_USE_ECA', 'True') == 'True'
         
        # 创建可配置模型
        model = ConfigurableUNet(
            out_channels=1,
            use_resnet=use_resnet,
            use_fpn=use_fpn,
            use_eca=use_eca,
            dropout_rate=0.2
        ).to(device)
        
        # 生成模型名称
        components = []
        if use_resnet:
            components.append("ResNet50")
        else:
            components.append("Baseline")
        if use_fpn:
            components.append("FPN")
        if use_eca:
            components.append("ECA")
        
        model_name = "+".join(components) if components else "Pure"
        
        logging.info(f"使用模型配置: {model_name}")
        logging.info(f"  - 编码器: {'ResNet50 (预训练)' if use_resnet else 'Baseline卷积'}")
        logging.info(f"  - FPN: {'启用' if use_fpn else '禁用'}")
        logging.info(f"  - ECA注意力: {'启用' if use_eca else '禁用'}")

        # 添加权重衰减以增强正则化
        weight_decay = 1e-4

        # 恢复为组合损失（BCE+Dice+Focal）
        criterion = CombinedLoss(alpha=0.4, beta=0.4)

        # 使用更小的初始学习率并添加权重衰减
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=weight_decay)

        # 使用更平滑的学习率调度器，增加T_max周期
        scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

        # 训练模型
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            100, device, save_dir, scheduler,
        )

        logging.info("模型训练完成！")

        # 加载最佳模型进行测试
        best_model_path = save_dir / 'best_unet.pth'
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # 评估参数：r基准与敏感性阈值列表
        distance_threshold = 40.0
        sweep_thresholds = [10, 15, 20, 25, 30, 35, 40]

        # 在训练集上评估（输出与测试集相同的指标字段）
        logging.info("在训练集上评估模型...")
        train_results, train_summary = evaluate_and_visualize(
            image_dir, annotation_dir, save_dir / 'train_results', model, device,
            sample_files=split_files['train'],
            distance_threshold=distance_threshold,
            sweep_thresholds=sweep_thresholds,
        )

        # 保存训练集评估结果
        with open(save_dir / "train_evaluation_summary.json", 'w') as f:
            json.dump(train_summary, f, indent=4)

        # 在测试集上评估
        logging.info("在测试集上评估模型...")
        test_results, test_summary = evaluate_and_visualize(
            image_dir, annotation_dir, save_dir / 'test_results', model, device,
            sample_files=split_files['test'],
            distance_threshold=distance_threshold,
            sweep_thresholds=sweep_thresholds,
        )

        # 保存测试集评估结果
        with open(save_dir / "test_evaluation_summary.json", 'w') as f:
            json.dump(test_summary, f, indent=4)

        # 输出训练集结果
        logging.info("="*60)
        logging.info("训练集最终评估指标:")
        logging.info("="*60)
        logging.info(f"AP: {train_summary['AP']:.4f}")
        logging.info(f"准确率 (Precision): {train_summary['avg_precision']:.4f}")
        logging.info(f"召回率 (Recall): {train_summary['avg_recall']:.4f}")
        logging.info(f"F1分数: {train_summary['avg_f1']:.4f}")
        logging.info(f"MAE: {train_summary['MAE']:.2f}")
        logging.info(f"FPS: {train_summary['FPS']:.2f}")
        logging.info(f"Prediction speed/per image: {train_summary['prediction_speed_per_image_ms']:.2f}ms")
        logging.info(f"总预测数: {train_summary['pred_total']}")
        logging.info(f"总真实数: {train_summary['true_total']}")
        logging.info(f"评估图像数: {train_summary['total_images']}")

        # 输出测试集结果
        logging.info("="*60)
        logging.info("测试集最终评估指标:")
        logging.info("="*60)
        logging.info(f"AP: {test_summary['AP']:.4f}")
        logging.info(f"准确率 (Precision): {test_summary['avg_precision']:.4f}")
        logging.info(f"召回率 (Recall): {test_summary['avg_recall']:.4f}")
        logging.info(f"F1分数: {test_summary['avg_f1']:.4f}")
        logging.info(f"MAE: {test_summary['MAE']:.2f}")
        logging.info(f"FPS: {test_summary['FPS']:.2f}")
        logging.info(f"Prediction speed/per image: {test_summary['prediction_speed_per_image_ms']:.2f}ms")
        logging.info(f"总预测数: {test_summary['pred_total']}")
        logging.info(f"总真实数: {test_summary['true_total']}")
        logging.info(f"评估图像数: {test_summary['total_images']}")
        
        # 输出对比结果
        logging.info("="*60)
        logging.info("训练集 vs 测试集 对比:")
        logging.info("="*60)
        logging.info(f"{'指标':<25} {'训练集':<15} {'测试集':<15} {'差值':<15}")
        logging.info("-" * 70)
        logging.info(f"{'AP':<25} {train_summary['AP']:<15.4f} {test_summary['AP']:<15.4f} {train_summary['AP'] - test_summary['AP']:<15.4f}")
        logging.info(f"{'Precision':<25} {train_summary['avg_precision']:<15.4f} {test_summary['avg_precision']:<15.4f} {train_summary['avg_precision'] - test_summary['avg_precision']:<15.4f}")
        logging.info(f"{'Recall':<25} {train_summary['avg_recall']:<15.4f} {test_summary['avg_recall']:<15.4f} {train_summary['avg_recall'] - test_summary['avg_recall']:<15.4f}")
        logging.info(f"{'F1':<25} {train_summary['avg_f1']:<15.4f} {test_summary['avg_f1']:<15.4f} {train_summary['avg_f1'] - test_summary['avg_f1']:<15.4f}")
        logging.info(f"{'MAE':<25} {train_summary['MAE']:<15.2f} {test_summary['MAE']:<15.2f} {train_summary['MAE'] - test_summary['MAE']:<15.2f}")
        logging.info(f"{'FPS':<25} {train_summary['FPS']:<15.2f} {test_summary['FPS']:<15.2f} {train_summary['FPS'] - test_summary['FPS']:<15.2f}")
        logging.info(f"{'Prediction speed (ms)':<25} {train_summary['prediction_speed_per_image_ms']:<15.2f} {test_summary['prediction_speed_per_image_ms']:<15.2f} {train_summary['prediction_speed_per_image_ms'] - test_summary['prediction_speed_per_image_ms']:<15.2f}")
        logging.info("="*70)

    except Exception as e:
        logging.error(f"程序执行出错 {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
