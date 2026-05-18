import torch
import torch.nn as nn
import torchvision.models as models

# class PiVisionNet(nn.Module):
#     def __init__(self, backbone_name="resnet18", pretrained=True, hidden=256, out_agents=2):
#         super().__init__()
#         self.backbone_name = backbone_name
#         backbone = getattr(models, backbone_name)(pretrained=pretrained)
#         in_features = backbone.fc.in_features
#         backbone.fc = nn.Identity()
#         self.backbone = backbone
#         self.out_agents = out_agents

#         self.fc1 = nn.Linear(in_features, hidden)
#         self.act = nn.ReLU(inplace=True)
#         # self.head = nn.Linear(hidden, 2 * 4)  # 2 agents * (dist,cos,sin,exist)
#         self.head = nn.Linear(hidden, out_agents * 3)  # 2 agents * (x,y,exist)

#     def forward(self, x):
#         feats = self.backbone(x)
#         h = self.act(self.fc1(feats))
#         out = self.head(h)                # [B, 8]
#         # out = out.view(-1, 2, 4)          # [B, 2, 4]
#         out = out.view(-1, self.out_agents, 3)          # [B, 2, 3]
#         # out[...,0]=z_dist, out[...,1]=z_cos, out[...,2]=z_sin, out[...,3]=logit_exist
#         return out
    
try:
    # new weights API (torchvision >= 0.13)
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models import MobileNet_V3_Small_Weights
    HAS_NEW_WEIGHTS = True
except Exception:
    HAS_NEW_WEIGHTS = False


class PiVisionNet(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True, hidden=256, out_agents=2):
        super().__init__()
        self.backbone_name = backbone_name
        self.out_agents = out_agents

        if backbone_name.lower() in ["resnet18", "resnet-18"]:
            # --- ResNet18 (your original path) ---
            if pretrained and hasattr(models, "ResNet18_Weights"):
                backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                backbone = models.resnet18(pretrained=pretrained)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()  # produce a 512-d vector
            self.backbone = backbone

        elif backbone_name.lower() in ["mobilenet_v3_large", "mobilenetv3_large", "mnetv3l"]:
            # --- MobileNetV3-Large ---
            if HAS_NEW_WEIGHTS and pretrained:
                backbone = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            else:
                backbone = models.mobilenet_v3_large(pretrained=pretrained)

            # MobileNetV3-Large forward is: features -> avgpool -> flatten -> classifier
            # classifier = [Linear(960->1280), Hardswish, Dropout, Linear(1280->num_classes)]
            # Replace ONLY the last Linear so the network outputs a 1280-d feature vector.
            in_features = backbone.classifier[-1].in_features  # 1280
            backbone.classifier[-1] = nn.Identity()
            self.backbone = backbone
            
        elif backbone_name.lower() in ["mobilenet_v3_small", "mobilenetv3_small", "mnetv3s"]:
            # --- MobileNetV3-Large ---
            if HAS_NEW_WEIGHTS and pretrained:
                backbone = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            else:
                backbone = models.mobilenet_v3_small(pretrained=pretrained)

            # MobileNetV3-Large forward is: features -> avgpool -> flatten -> classifier
            # classifier = [Linear(960->1280), Hardswish, Dropout, Linear(1280->num_classes)]
            # Replace ONLY the last Linear so the network outputs a 1280-d feature vector.
            in_features = backbone.classifier[-1].in_features  # 1280
            backbone.classifier[-1] = nn.Identity()
            self.backbone = backbone

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # --- Your head ---
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU(inplace=True)
        self.head = nn.Linear(hidden, out_agents * 3)  # (x, y, exist) per agent

    def forward(self, x):
        feats = self.backbone(x)                 # [B, 512] for ResNet18, [B,1280] for MNetV3-L
        h = self.act(self.fc1(feats))
        out = self.head(h)                       # [B, out_agents*3]
        out = out.view(-1, self.out_agents, 3)   # [B, A, 3]
        return out


class AreaAttentionNet(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True, num_channels=1, heat_shape=(100,100)):
        super().__init__()
        
        # --- ResNet18 backbone ---
        # resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # keep only conv layers (drop avgpool + fc)
        # self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # (B,512,H/32,W/32)
        
        self.backbone_name = backbone_name
        
        if backbone_name.lower() in ["resnet18", "resnet-18"]:
            # --- ResNet18 (your original path) ---
            if pretrained and hasattr(models, "ResNet18_Weights"):
                backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                backbone = models.resnet18(pretrained=pretrained)
                
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # (B,512,H/32,W/32)
            backbone_out_ch = 512

        elif backbone_name.lower() in ["mobilenet_v3_large", "mobilenetv3_large", "mnetv3l"]:
            # --- MobileNetV3-Large ---
            if HAS_NEW_WEIGHTS and pretrained:
                backbone = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            else:
                backbone = models.mobilenet_v3_large(pretrained=pretrained)
                
            self.backbone = backbone.features            # (B,960,H/32,W/32)
            backbone_out_ch = 960
            
        else:
            raise(Exception("backbone invalid"))
        
        # --- Upsampling head ---
        # Input 128x128 -> ResNet output 512 x 4 x 4
        # We want 100x100 output
        self.head = nn.Sequential(
            nn.ConvTranspose2d(backbone_out_ch, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 128x128
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_channels, kernel_size=1)                         # 128x128
        )
        
        # final resize to 100x100 (bilinear interpolation)
        self.upsample = nn.Upsample(size=heat_shape, mode='bilinear', align_corners=False)

    def forward(self, x):
        feat = self.backbone(x)       # (B,512,4,4)
        heat = self.head(feat)        # (B,C,64,64)
        heat = self.upsample(heat)    # (B,C,100,100)
        return heat