import torch.nn as nn
import torch

class MultiScaleCNN(nn.Module):
    def __init__(self, max_len, num_classes=3):
        super().__init__()
        self.embed = nn.Embedding(21, 64, padding_idx=0)  # 20AA + padding
        self.branch3 = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1, groups=32),#局部特征
            nn.Conv1d(32, 32, 1),#特征关联
            nn.BatchNorm1d(32),#归一化
            nn.ReLU(),#激活分类
            nn.Dropout(0.5),#防止过度拟合
            nn.AdaptiveAvgPool1d(max_len)  # 平均池化，调整长度
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(64, 32, 5, padding=2, dilation=3, groups=32),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(max_len)  # 调整长度
        )
        # 3. 特征融合
        self.fusion = nn.Sequential(
            nn.Conv1d(64, 64, 1),  # 1x1卷积整合特征
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(max_len)  # 保证输出长度一致
        )
        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Conv1d(64, num_classes, 1),
            nn.Flatten(start_dim=2)  # (batch, seq_len, num_classes)
        )
    
    def forward(self, x):
        # 输入形状: (batch, seq_len)
        x = self.embed(x)  # (batch, seq_len, 64)
        x = x.permute(0, 2, 1)  # (batch, 64, seq_len)
        
        # 多尺度特征提取
        x3 = self.branch3(x)  # (batch, 32, seq_len)
        x5 = self.branch5(x)   # (batch, 32, seq_len)
        x = torch.cat([x3, x5], dim=1)  # (batch, 64, seq_len)
        
        # 特征融合
        x = self.fusion(x)  # (batch, 64, max_len)
        
        # 分类输出
        return self.classifier(x).permute(0, 2, 1)  # (batch, max_len, num_classes)