import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 128, normalize=False)
        self.down2 = UNetDown(128, 256)
        self.down3 = UNetDown(256, 512)
        self.down4 = UNetDown(512, 1024, dropout=0.5)
        self.down5 = UNetDown(1024, 1024, dropout=0.5)
        self.down6 = UNetDown(1024, 1024, dropout=0.5)
        self.down7 = UNetDown(1024, 1024, dropout=0.5)
        self.down8 = UNetDown(1024, 1024, dropout=0.5)
        self.down9 = UNetDown(1024, 1024, normalize=False, dropout=0.5)

        self.up1 = UNetUp(1024, 1024, dropout=0.5)
        self.up2 = UNetUp(2048, 1024, dropout=0.5)
        self.up3 = UNetUp(2048, 1024, dropout=0.5)
        self.up4 = UNetUp(2048, 1024, dropout=0.5)
        self.up5 = UNetUp(2048, 1024, dropout=0.5)
        self.up6 = UNetUp(2048, 512)
        self.up7 = UNetUp(1024, 256)
        self.up8 = UNetUp(512, 128)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, out_channels, 4, padding=1),   
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        d9 = self.down9(d8)

        u1 = self.up1(d9)
        u2 = self.up2(torch.cat([u1, d8], dim=1))
        u3 = self.up3(torch.cat([u2, d7], dim=1))
        u4 = self.up4(torch.cat([u3, d6], dim=1))
        u5 = self.up5(torch.cat([u4, d5], dim=1))
        u6 = self.up6(torch.cat([u5, d4], dim=1))
        u7 = self.up7(torch.cat([u6, d3], dim=1))
        u8 = self.up8(torch.cat([u7, d2], dim=1))

        return self.final(torch.cat([u8, d1], dim=1))


#############################
#      Discriminator
#############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=15):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 128, normalization=False),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 1024),
            *discriminator_block(1024, 1024),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(1024, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat([img_A, img_B], dim=1)
        x = self.model(img_input)
        return x
