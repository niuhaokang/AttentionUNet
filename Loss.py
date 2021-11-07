import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor,device=None):
        assert device
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.device = device
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size(),device=self.device).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size(),device=self.device).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self,device):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class GANLogisticLoss(nn.Module):
    def __init__(self,mode):
        super().__init__()
        self.mode = mode
    def forward(self,list_pred):
        if self.mode == 'D':
            assert len(list_pred) == 2
            real_loss = F.softplus(-list_pred[0])
            fake_loss = F.softplus(list_pred[1])
            return real_loss.mean() + fake_loss.mean()
        elif self.mode == 'G':
            assert len(list_pred) == 1
            return F.softplus(-list_pred[0]).mean()
        else:
            print("This is a unknown mode of {}".format(self.mode))
            assert 1 > 2

class IDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load('/home/haokang/hk/hkGAN/networks/ArcFace/model_ir_se50.pth'))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self,x,y,y_hat):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0

        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append({'diff_target': float(diff_target),
                            'diff_input': float(diff_input),
                            'diff_views': float(diff_views)})
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count


class SemiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sim = nn.CosineSimilarity()
        self.sfm = nn.Softmax(dim=0)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    def forward(self, supervisied_A_feature,
                unsupervisied_A_feature,
                supervisied_B_feature,
                unsupervisied_B_feature):

        A_sim = self.sim(supervisied_A_feature, unsupervisied_A_feature)
        B_sim = self.sim(supervisied_B_feature, unsupervisied_B_feature)
        A_dist = self.sfm(A_sim)
        B_dist = self.sfm(B_sim)

        rel_loss = 1000 * \
                   self.kl_loss(torch.log(A_dist), B_dist)  # distance consistency loss
        return rel_loss

if __name__ == '__main__':
    # from networks.Pix2PixHD.Discriminator import MultiscaleDiscriminator
    # from utils import get_norm_layer as get_norm_layer
    # from utils import weights_init as weights_init
    import os
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    device = 'cuda:0'
    # GANLoss = GANLoss(use_lsgan=True,tensor=torch.cuda.FloatTensor)
    # D = MultiscaleDiscriminator(input_nc=6,
    #                             ndf=64,
    #                             n_layers=3,
    #                             norm_layer=get_norm_layer(norm_type='instance'),
    #                             use_sigmoid=False,
    #                             num_D=2,
    #                             getIntermFeat=False).to(device)
    # D.apply(weights_init)
    # x = torch.rand([3, 3, 256, 256]).to(device)
    # x_ = torch.rand([3, 3, 256, 256]).to(device)
    # y = D(torch.cat([x, x_], dim=1))
    # z = GANLoss(y,True)
    # print(z)

    x = torch.rand([256, 3]).to(device)
    x_ = torch.rand([256, 3]).to(device)
    y = torch.rand([256, 3]).to(device)
    y_ = torch.rand([256, 3]).to(device)

    loss = SemiLoss().to(device)
    loss(x,x_,y,y_)