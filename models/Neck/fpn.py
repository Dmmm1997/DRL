import torch
import torch.nn as nn
from ..utils import weight_init


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        inputs2 = self.up(self.conv1(inputs2))
        outputs = inputs2 +inputs1
        # outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv2(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self,  C3_size, C4_size, C5_size, feature_size=384):
        super(Unet, self).__init__()
        in_filters = [64, 128, 320]
        out_filters = [64, 64, 128]
        # upsampling
        self.up_concat2 = unetUp(in_filters[2], out_filters[2])
        # self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[1], out_filters[1])

        # final conv (without any concat)
        # self.final=nn.Conv2d(out_filters[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.final = nn.Conv2d(out_filters[0], feature_size, 1)

    def forward(self, inputs):
        feat1, feat2, feat3 = inputs

        up2 = self.up_concat2(feat2, feat3)
        final = self.up_concat1(feat1, up2)

        # final = self.final(up1)

        return final

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()



class FPN_I4(nn.Module):
    def __init__(self, input_dims,  output_dims=384, **kwargs):  # 384 1
        super(FPN_I4, self).__init__()

        C3_size, C4_size, C5_size, C6_size = input_dims
        self.P6_1 = nn.Conv2d(C6_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.P6_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P6_2 = nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1)

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(output_dims, output_dims, kernel_size=3, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(output_dims, output_dims, kernel_size=3, padding=1)

    def forward(self, inputs):
        C3, C4, C5, C6 = inputs

        P6_x = self.P6_1(C6)
        P6_upsampled_x = self.P6_upsampled(P6_x)
        P6_x = self.P6_2(P6_x)

        P5_x = self.P5_1(C5)
        P5_x = P6_upsampled_x + P5_x
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        

        return [P3_x, P4_x, P5_x, P6_x]


class FPN_I3(nn.Module):
    def __init__(self, input_dims, output_dims=384, **kwargs):  # 384 1
        super(FPN_I3, self).__init__()
        
        C3_size, C4_size, C5_size = input_dims
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(output_dims, output_dims, kernel_size=3, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(output_dims, output_dims, kernel_size=3, padding=1)

        # # "P6 is obtained via a 3x3 stride-2 conv on C5"
        # self.P6 = nn.Conv2d(C5_size, output_dims, kernel_size=3, stride=2, padding=1)

        # # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        # self.P7_1 = nn.ReLU()
        # self.P7_2 = nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=2, padding=1)

        # self.apply(weight_init)


    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        # P6_x = self.P6(C5)
        # # panet
        # P7_x = self.P7_1(P6_x)
        # P7_x = self.P7_2(P7_x)
        return [P3_x, P4_x, P5_x]
    

class PANET_fusion(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=1):  # 384 1
        super(PANET_fusion, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.convN = nn.Conv2d(1, 1, 3, 2, 1)
        self.relu = nn.ReLU(inplace=True)
        # self.gelu = nn.GELU()

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)
        # panet
        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        N2 = P3_x
        N2_ = self.convN(N2)
        N2_ = self.relu(N2_)
        #
        N3 = N2_ + P4_x
        #
        N3_ = self.convN(N3)
        N3_ = self.relu(N3_)

        N4 = N3_ + P5_x
        return [N2, N3, N4]


class PANET(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=384):  # 384 1
        super(PANET, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.convN = nn.Conv2d(feature_size, feature_size, 3, 2, 1)
        # self.convN2 = nn.Conv2d(feature_size, feature_size, 3, 2, 1)
        self.relu = nn.ReLU(inplace=True)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.gelu = nn.GELU()

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)
        # panet
        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        N2 = P3_x
        N2_ = self.convN(N2)
        N2_ = self.relu(N2_)
        #
        N3 = N2_ + P4_x
        #
        N3_ = self.convN(N3)
        N3_ = self.relu(N3_)

        N4 = N3_ + P5_x
        return [N2, N3, N4]


class PANET_conver(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=300):  # 384 1
        super(PANET_conver, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_c = nn.Conv2d(feature_size * 2, feature_size, kernel_size=1, stride=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_c = nn.Conv2d(feature_size * 2, feature_size, kernel_size=1, stride=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.convN = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.gelu = nn.GELU()

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = torch.concat((P5_upsampled_x, P4_x), 1)
        P4_x = self.P4_c(P4_x)
        # P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        # P4_x = self.P4_2(P4_x)  # be replase by P4_c

        P3_x = self.P3_1(C3)
        P3_x = torch.concat((P4_upsampled_x, P3_x), 1)
        P3_x = self.P3_c(P3_x)
        # P3_x = P3_x + P4_upsampled_x
        # P3_x = self.P3_2(P3_x)  # be replase by P4_c

        P6_x = self.P6(C5)
        # panet
        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        N2 = P3_x
        N2_ = self.convN(N2)  # down samplle
        N2_ = self.relu(N2_)
        #
        N3 = N2_ + P4_x
        #
        N3_ = self.convN(N3)  # down samplle
        N3_ = self.relu(N3_)

        N4 = N3_ + P5_x
        return [N2, N3, N4]


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[3, 5, 7]):
        super(SpatialPyramidPooling, self).__init__()
        self.conv1 = nn.Conv2d(320, 160, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(640, 320, kernel_size=1, stride=1, bias=False)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, z):
        x = self.conv1(z[2])
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        # features = torch.cat(features + [x], dim=1)
        features = features[0]+features[1]+features[2]+x
        z[2] = features

        return z
