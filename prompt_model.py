import torch
import torch.nn as nn
import os
import torchvision.models as models

class Prompt(nn.Module):
    def __init__(self):
        super(Prompt, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), # [, 64, 224, 224]
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), # [, 64, 224, 224]
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # [, 64, 112, 112]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # [, 128, 112, 112]
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), # [, 128, 112, 112]
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=1, padding=1), # [, 256, 112, 112]
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # [, 256, 56, 56]
            nn.BatchNorm2d(256),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1), # [, 256, 56, 56]
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1), # [, 128, 112, 112]
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1), # [, 64, 112, 112]
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1), # [, 32, 112, 112]
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1), # [, 32, 112, 112]
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # [, 16, 224, 224]
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1), # [, 3, 224, 224]
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class Prompt_light(nn.Module):
    def __init__(self):
        super(Prompt_light, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), # [, 64, 96, 96]
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), # [, 64, 224, 224]
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # [, 64, 112, 112]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # [, 128, 112, 112]
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=1, padding=1), # [, 256, 112, 112]
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # [, 256, 56, 56]
            nn.BatchNorm2d(256),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1), # [, 256, 56, 56]
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1), # [, 128, 112, 112]
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1), # [, 64, 112, 112]
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1), # [, 32, 112, 112]
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # [, 16, 224, 224]
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1), # [, 3, 224, 224]
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class P_Model(nn.Module):
    def __init__(self, args):
        super(P_Model, self).__init__()
        if args['prompt'] == 'prompt':
            self.prompt = Prompt()
        elif args['prompt'] == 'light':
            self.prompt = Prompt_light()
        if args['backbond'] == 'rn50':
            self.backbond = models.resnet50(pretrained=True)
        elif args['backbond'] == 'rn101':
            self.backbond = models.resnet101(pretrained=True)
        for param in self.backbond.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(1000, 100)
    def forward(self, x):
        prompt = self.prompt(x)
        input = prompt + x
        output = self.backbond(input)
        output = self.fc(output)
        return output

# if __name__ == '__main__':
#     x = torch.randn((1, 3, 224, 224))
#     model = P_Model()
#     for name, param in model.named_parameters():
#         print(name, param.requires_grad)