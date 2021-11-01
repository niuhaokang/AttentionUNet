import torch.nn as nn
import torch
import torch.nn.functional as F
import functools
from model.transformer import MultiHeadAttentionOne
from model.RRDBNet_arch import RRDBNet

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class DimTransform(nn.Module):
    def __init__(self, chan):
        super(DimTransform, self).__init__()
        model = [
            nn.ConvTranspose2d(chan, chan, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(chan),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, feat):
        return self.model(feat)



class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        self.transformer = MultiHeadAttentionOne(1, 1024, 1024, 1024, dropout=0.5)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)
        self.transformer = MultiHeadAttentionOne(1, 1024, 1024, 1024, dropout=0.5)
        self.dim_transform = nn.Sequential(*[
            nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
        ])

    def extract(self, feat):

        norm_feat = F.normalize(feat, dim=1)
        norm_feat = self.dim_transform(norm_feat)
        norm_feat = norm_feat.flatten(2, 3)

        return feat, norm_feat


    def forward(self, supervisied_img, unsupervisied_img, alpha=0.7):
        B, _, H, W = supervisied_img.size()
        supervisied_feat = self.model[:16](supervisied_img) # b * 1024 * 16 * 16
        unsupervisied_feat = self.model[:16](unsupervisied_img)  # b * 1024 * 16 * 16

        supervisied_v, q = self.extract(supervisied_feat) # b * 1024 * 16 * 16, b * 1024 * 256
        unsupervisied_v, k = self.extract(unsupervisied_feat)

        f_l = self.transformer(q, supervisied_v, supervisied_v)
        f_u = self.transformer(k, unsupervisied_v, unsupervisied_v)

        pred_l = torch.matmul(f_u, supervisied_v.flatten(2, 3)).view(B, 1024, 16, 16)
        pred_u = torch.matmul(f_l, unsupervisied_v.flatten(2, 3)).view(B, 1024, 16, 16)

        supervisied_img_ = self.model[16:](pred_l)
        unsupervisied_img_ = self.model[16:](pred_u)

        return supervisied_img_, unsupervisied_img_, f_l, f_u

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    device = 'cuda'


    G = GlobalGenerator(input_nc=3,
                        output_nc=3,
                        ngf=64,
                        n_downsampling=4,
                        n_blocks=9,
                        norm_layer=get_norm_layer(norm_type='instance')).to(device)
    G.apply(weights_init)

    a = torch.rand((3, 3, 256, 256)).to(device)
    b = torch.rand((3, 3, 256, 256)).to(device)
    out,_,_,_ = G(a, b)
    print(out.size())
    # print("sss")
    # print(out.size())
    # sfm = nn.Softmax(dim=1)
    # kl_loss = nn.KLDivLoss()
    # sim = nn.CosineSimilarity()
