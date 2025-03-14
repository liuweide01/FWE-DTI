import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DModel(nn.Module):
    def __init__(self, in_channels=28):
        super(Conv3DModel, self).__init__()
        # --- Block 1 ---
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.conv1_1 = nn.Conv3d(32, 32, kernel_size=1, padding=0)
        self.conv1_2 = nn.Conv3d(32, 32, kernel_size=1, padding=0)

        # --- Block 2 ---
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv3d(32, 32, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv3d(32, 32, kernel_size=1, padding=0)

        # --- Block 3 ---
        self.conv3 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv3d(32, 32, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv3d(32, 32, kernel_size=1, padding=0)

        # --- Block 4 ---
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv3d(32, 32, kernel_size=1, padding=0)
        self.conv4_2 = nn.Conv3d(32, 32, kernel_size=1, padding=0)

        # --- Block 5 ---
        self.conv5 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv3d(32, 32, kernel_size=1, padding=0)
        self.conv5_2 = nn.Conv3d(32, 32, kernel_size=1, padding=0)

        self.conv6 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv6_1 = nn.Conv3d(32, 32, kernel_size=1, padding=0)
        self.conv6_2 = nn.Conv3d(32, 32, kernel_size=1, padding=0)

        # --- Block 4 ---
        self.conv7 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv7_1 = nn.Conv3d(32, 32, kernel_size=1, padding=0)
        self.conv7_2 = nn.Conv3d(32, 32, kernel_size=1, padding=0)

        # --- Block 5 ---
        self.conv8 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv8_1 = nn.Conv3d(32, 32, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv3d(32, 32, kernel_size=1, padding=0)


        # --- Block 4 ---
        self.conv9 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv9_1 = nn.Conv3d(32, 32, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv3d(32, 32, kernel_size=1, padding=0)

        # --- Block 5 ---
        self.conv10 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv10_1 = nn.Conv3d(32, 32, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv3d(32, 32, kernel_size=1, padding=0)

        # Decoder layers
        self.decoder_0 = nn.Conv3d(32, 32, kernel_size=1, padding=0)
        self.decoder   = nn.Conv3d(32, 7,   kernel_size=1, padding=0)

    def forward(self, x):
        # --- Block 1 ---
        x1 = F.relu(self.conv1(x))                  # 3×3×3 conv
        x1_bottleneck = F.relu(self.conv1_2(
                                F.relu(self.conv1_1(x1)) ))
        x1_out = x1 + x1_bottleneck                 # skip connection

        # --- Block 2 ---
        x2 = F.relu(self.conv2(x1_out))             # 3×3×3 conv
        x2_bottleneck = F.relu(self.conv2_2(
                                F.relu(self.conv2_1(x2)) ))
        x2_out = x2 + x2_bottleneck                 # skip connection

        # --- Block 3 ---
        x3 = F.relu(self.conv3(x2_out))             # 3×3×3 conv
        x3_bottleneck = F.relu(self.conv3_2(
                                F.relu(self.conv3_1(x3)) ))
        x3_out = x3 + x3_bottleneck                 # skip connection

        # --- Block 4 ---
        x4 = F.relu(self.conv4(x3_out))             # 3×3×3 conv
        x4_bottleneck = F.relu(self.conv4_2(
                                F.relu(self.conv4_1(x4)) ))
        x4_out = x4 + x4_bottleneck                 # skip connection


        # --- Block 2 ---
        x5 = F.relu(self.conv5(x4_out))             # 3×3×3 conv
        x5_bottleneck = F.relu(self.conv5_2(
                                F.relu(self.conv5_1(x5)) ))
        x5_out = x5 + x5_bottleneck                 # skip connection


        # --- Block 2 ---
        x6 = F.relu(self.conv6(x5_out))             # 3×3×3 conv
        x6_bottleneck = F.relu(self.conv6_2(
                                F.relu(self.conv6_1(x6)) ))
        x6_out = x6 + x6_bottleneck                 # skip connection


        # --- Block 2 ---
        x7 = F.relu(self.conv7(x6_out))             # 3×3×3 conv
        x7_bottleneck = F.relu(self.conv7_2(
                                F.relu(self.conv7_1(x7)) ))
        x7_out = x7 + x7_bottleneck                 # skip connection

        # --- Block 2 ---
        x8 = F.relu(self.conv8(x7_out))             # 3×3×3 conv
        x8_bottleneck = F.relu(self.conv8_2(
                                F.relu(self.conv8_1(x8)) ))
        x8_out = x8 + x8_bottleneck                 # skip connection

        # --- Block 2 ---
        x9 = F.relu(self.conv9(x8_out))             # 3×3×3 conv
        x9_bottleneck = F.relu(self.conv9_2(
                                F.relu(self.conv9_1(x9)) ))
        x9_out = x9 + x9_bottleneck                 # skip connection

        # --- Block 2 ---
        x10 = F.relu(self.conv10(x9_out))             # 3×3×3 conv
        x10_bottleneck = F.relu(self.conv10_2(
                                F.relu(self.conv10_1(x10)) ))
        x10_out = x10 + x10_bottleneck                 # skip connection


        # Decoder
        d0 = F.relu(self.decoder_0(x10_out))
        out = self.decoder(d0)
        return out

# -----------------------------------------------------------
# Example test
if __name__ == "__main__":
    model = Conv3DModel(in_channels=28)

    # Suppose you have a batch_size=1, 28 channels, volume of size 64×64×64
    input_tensor = torch.rand(1, 28, 64, 64, 64)
    output = model(input_tensor)
    print("Output shape:", output.shape)
    # Expect: [1, 7, 64, 64, 64]
