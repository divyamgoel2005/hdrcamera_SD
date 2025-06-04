# cnn_preprocess.py
import torch
import torch.nn as nn

class PreprocessCNN(nn.Module):
    def __init__(self):
        super(PreprocessCNN, self).__init__()

        # 1x1 convolution to slightly scale RGB channels
        self.enhance = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, bias=True),
            nn.BatchNorm2d(3)
        )

        # Initialize to identity + slight contrast boost
        with torch.no_grad():
            self.enhance[0].weight.copy_(torch.eye(3).view(3, 3, 1, 1) * 1.05)  # slight boost
            self.enhance[0].bias.zero_()

    def forward(self, x):
        return torch.clamp(self.enhance(x), 0, 1)
