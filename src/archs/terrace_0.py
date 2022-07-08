from constants import *
from utils import util
softmax = nn.Softmax(dim=2)

def mid_conv_layer(depth_in, depth_out):
    return nn.Sequential(
                nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(depth_out),
                nn.ReLU(),
            )

def deconv_layer(depth_in, depth_out, output_size=-1):
    if output_size == -1:
        return nn.Sequential(
                    nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
            )
    else:
        return nn.Sequential(
                    nn.Conv2d(depth_in, depth_out, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU(),
                    nn.Upsample(size=output_size, mode='bilinear'),
            )

class Arch(nn.Module):
    def __init__(self, num_contour=5):
        super(Arch, self).__init__()
        self.name = 'TERRACE_v0'
        #resnet hyper params
        RESNET_DEPTH_3 = 512
        RESNET_DEPTH_4 = 1024
        RESNET_DEPTH_5 = 2048
        CONTOUR_DEPTH = num_contour + 1

        resnet_model = torchvision.models.resnet50(pretrained=True)

        self.conv123 = nn.Sequential(*list(resnet_model.children())[0:6]) #0
        self.conv4 = nn.Sequential(*list(resnet_model.children())[6]) #1
        self.conv5 = nn.Sequential(*list(resnet_model.children())[7]) #2

        self.mid3 = mid_conv_layer(RESNET_DEPTH_3, CONTOUR_DEPTH) #3
        self.mid4 = mid_conv_layer(RESNET_DEPTH_4, CONTOUR_DEPTH) #4
        self.mid5 = mid_conv_layer(RESNET_DEPTH_5, CONTOUR_DEPTH)  #child 5

        self.deconv_5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear')) #child 6
        self.deconv_4 = deconv_layer(CONTOUR_DEPTH * 2, CONTOUR_DEPTH) #7
        self.deconv_3 = deconv_layer(CONTOUR_DEPTH * 2, CONTOUR_DEPTH, output_size=256) #8

        #counting
        self.AvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1)) #9
        self.fc = nn.Linear(in_features=2048, out_features=1, bias=True) #10


    def forward(self, x):
        #convolution layers
        conv_out3 = self.conv123(x)
        mid_out3 = self.mid3(conv_out3)
        conv_out4 = self.conv4(conv_out3)
        mid_out4 = self.mid4(conv_out4)
        conv_out5 = self.conv5(conv_out4)
        mid_out5 = self.mid5(conv_out5)

        #deconvolution layers
        deconv_out5 = self.deconv_5(mid_out5)
        
        fused4 = torch.cat([deconv_out5, mid_out4], 1)
        deconv_out4 = self.deconv_4(fused4)

        fused3 = torch.cat([deconv_out4, mid_out3], 1)
        deconv_out3 = self.deconv_3(fused3)

        #attention layer
        (BATCH_SIZE, DEPTH, SIZE5, SIZE5) = mid_out5.size()
        hA = mid_out5.view(BATCH_SIZE, DEPTH, -1)                 #(B, D, 49)
        hA = hA.transpose(1, 2)                 #(B, 49, D)
        hA = torch.unsqueeze(hA, 2)            #(B, 49, 1, D)
        hA = hA.contiguous().view(-1, 1, DEPTH)             #(B * 49, 1, D)
        prob = softmax(hA)

        WEIGHT = torch.tensor([[0], [1], [1.2], [1.4], [1.6], [1.8]], dtype=torch.float).to(device)  #(D, 1)  #id = 2
        prob = torch.matmul(prob, WEIGHT) #(B * 49, 1, 1)

        #put attention layer on conv5 feature
        (BATCH_SIZE, DEPTH, _, _) = conv_out5.size()
        conv_feature = conv_out5.view(BATCH_SIZE, DEPTH, -1)         #(B, D, 49)
        conv_feature = conv_feature.transpose(1, 2)              #(B, 49, D)
        conv_feature = torch.unsqueeze(conv_feature, 2)          #(B, 49, 1, D)
        conv_feature = conv_feature.contiguous().view(-1, 1, DEPTH)   #(B * 49, 1, D)

        out_attention = torch.bmm(prob, conv_feature)
        out_attention = out_attention.view(BATCH_SIZE, SIZE5*SIZE5, 1, DEPTH)    #(B, 49, 1, D)
        out_attention = out_attention.squeeze(2)                          #(B, 49, D)           #after attention, it has same size as x_0
        out_attention = out_attention.transpose(1, 2)                #(B, D, 49)
        out_attention = out_attention.view(BATCH_SIZE, DEPTH, SIZE5, SIZE5)

        out = self.AvgPool2d(out_attention)  #conv_out5, out_attention
        out = torch.squeeze(out)
        out = self.fc(out)

  
        return deconv_out3, out


    