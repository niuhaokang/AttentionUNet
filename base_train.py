import argparse
import itertools

import torch
from torch.utils.data import DataLoader
from torchvision import utils
import os
from dataset import ImageDataSet
from dataset import TestImageDataSet

from model.Generator import GlobalGenerator
from model.Discriminator import MultiscaleDiscriminator
from model.Patch import PatchSampleF

from utils import get_norm_layer as get_norm_layer
from utils import weights_init as weights_init
from Loss import GANLoss
from Loss import VGGLoss
from Loss import SemiLoss

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sketch')
    parser.add_argument('--data_root', type=str, default='/data/haokang/Semi-Supervisied-DataSet/cartoon')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--resolution', type=int, default=256)

    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--save_img_epoch', type=int, default=1)
    parser.add_argument('--save_model_epoch', type=int, default=10)

    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.5)
    parser.add_argument('--lambda_feat', type=float, default=10.0)

    parser.add_argument('--distributed', type=bool, default=False)

    parser.add_argument('--norm', type=str, default='instance')
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=3)

    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--n_downsampling_G', type=int, default=4)
    parser.add_argument('--n_blocks_G', type=int, default=9)

    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--n_layers_D', type=int, default=3)
    parser.add_argument('--use_sigmoid', type=bool, default=False)
    parser.add_argument('--num_D', type=int, default=2)
    parser.add_argument('--getIntermFeat', type=bool, default=False)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    device = 'cuda:1'

    assert args.name
    if not os.path.exists('./result'):
        os.mkdir('./result')
    if not os.path.exists(os.path.join('./result', args.name)):
        os.mkdir(os.path.join('./result', args.name))
        os.mkdir(os.path.join('./result', args.name, 'imgs'))
        os.mkdir(os.path.join('./result', args.name, 'checkpoints'))
        os.mkdir(os.path.join('./result', args.name, 'loss_txt'))
        os.mkdir(os.path.join('./result', args.name, 'figure'))
        os.mkdir(os.path.join('./result', args.name, 'train_data'))

    args.result_imgs = os.path.join('./result', args.name, 'imgs')
    args.result_checkpoints = os.path.join('./result', args.name, 'checkpoints')
    args.result_loss_txt = os.path.join('./result', args.name, 'loss_txt')
    args.result_train_data = os.path.join('./result', args.name, 'train_data')

    dataset = ImageDataSet(data_root=args.data_root,
                           input_size=256,
                           open_Method='RGB')
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4)
    test_dataset = ImageDataSet(data_root=args.data_root,
                                input_size=256,
                                open_Method='RGB')
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4)

    # The Generator of Pix2PixHD, x -> y
    G = GlobalGenerator(input_nc=args.input_nc,
                        output_nc=args.output_nc,
                        ngf=args.ngf,
                        n_downsampling=args.n_downsampling_G,
                        n_blocks=args.n_blocks_G,
                        norm_layer=get_norm_layer(norm_type=args.norm)).to(device)
    G.apply(weights_init)

    # The Discriminator of Pix2PixHD is used to discriminate domainX
    D = MultiscaleDiscriminator(input_nc=args.input_nc,
                                 ndf=args.ndf,
                                 n_layers=args.n_layers_D,
                                 norm_layer=get_norm_layer(norm_type=args.norm),
                                 use_sigmoid=args.use_sigmoid,
                                 num_D=args.num_D,
                                 getIntermFeat=args.getIntermFeat).to(device)
    D.apply(weights_init)
    P = PatchSampleF().to(device)

    # Loss items
    criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor, device=device)
    criterionFeat = torch.nn.L1Loss()
    criterionVGG = VGGLoss(device=device)
    criterionl2 = torch.nn.MSELoss()
    criterionSemi = SemiLoss()

    G_optim = torch.optim.Adam(G.parameters(),
                               lr=args.lr,
                               betas=(args.beta1, args.beta2))
    D_optim = torch.optim.Adam(D.parameters(),
                               lr=args.lr,
                               betas=(args.beta1, args.beta2))


    for epoch in range(args.epoch):
        if epoch % args.save_img_epoch == 0 and epoch != 0:
            with torch.no_grad():
                for idx, test_data in enumerate(test_dataloader):
                    if idx >= 20:
                        break
                    this_epoch = os.path.join(args.result_imgs, 'epoch_' + str(epoch))
                    if not os.path.exists(this_epoch):
                        os.mkdir(this_epoch)

                    name = str(idx) + '.png'
                    supervisied_img = test_data['supervisied_img_A'].to(device)
                    unsupervised_img = test_data['unsupervised_img'].to(device)
                    supervisied_res, unsupervised_res, _, _ = G(supervisied_img, unsupervised_img)

                    target_img = torch.cat([unsupervised_img,
                                            unsupervised_res],
                                           dim=3)
                    target_dir = os.path.join(this_epoch, name)
                    utils.save_image(
                        target_img,
                        target_dir,
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )
        if epoch % args.save_model_epoch == 0:
            pass
        count = 0
        for idx, data in enumerate(dataloader):
            loss_dict = {}
            # 输入数据
            supervisied_A = data['supervisied_img_A'].to(device)
            supervisied_B = data['supervisied_img_B'].to(device)
            unsupervised_A = data['unsupervised_img'].to(device)

            supervisied_A2B, unsupervised_A2B, f_l, f_u = G(supervisied_A, unsupervised_A)

            # update D
            requires_grad(G, False)
            requires_grad(P, False)
            requires_grad(D, True)

            D_real_pred = D(supervisied_B)
            D_fake_pred_supervisied = D(supervisied_A2B)
            D_fake_pred_unsupervisied = D(unsupervised_A2B)

            D_loss = 1/4 * criterionGAN(D_fake_pred_supervisied, False) +\
                     1/4 * criterionGAN(D_fake_pred_unsupervisied, False) +\
                     1/2 * criterionGAN(D_real_pred, True)

            loss_dict['D_loss'] = D_loss

            D.zero_grad()
            D_loss.backward(retain_graph=True)
            D_optim.step()

            # update G
            requires_grad(G, True)
            requires_grad(P, True)
            requires_grad(D, False)

            D_real_pred = D(supervisied_B)
            D_fake_pred_supervisied = D(supervisied_A2B)
            D_fake_pred_unsupervisied = D(unsupervised_A2B)

            supervisied_feat_A, supervisied_ids = P([supervisied_A],
                                                    num_patches=512,
                                                    patch_ids=None)
            unsupervised_feat_A, unsupervised_ids = P([unsupervised_A],
                                                      num_patches=512,
                                                      patch_ids=None)
            supervisied_feat_B, _ = P([supervisied_A2B],
                                      num_patches=512,
                                      patch_ids=supervisied_ids)
            unsupervised_feat_B, _ = P([unsupervised_A2B],
                                       num_patches=512,
                                       patch_ids=unsupervised_ids)
            loss_Semi = criterionSemi(supervisied_feat_A[0],
                                      unsupervised_feat_A[0],
                                      supervisied_feat_B[0],
                                      unsupervised_feat_B[0])

            G_loss_adv = 1/2 * criterionGAN(D_fake_pred_supervisied, True) +\
                         1/2 * criterionGAN(D_fake_pred_unsupervisied, True)

            GAN_Feat = 0.0
            feat_weights = 4.0 / (args.n_layers_D + 1)
            D_weights = 1.0 / args.num_D
            for i in range(args.num_D):
                for j in range(len(D_fake_pred_supervisied[i])):
                    GAN_Feat += D_weights * feat_weights * \
                                       criterionFeat(D_fake_pred_supervisied[i][j],
                                                     D_real_pred[i][j].detach()) * args.lambda_feat

            loss_VGG = criterionVGG(supervisied_A2B, supervisied_B) * args.lambda_feat

            loss_f = criterionl2(f_l, f_u)

            loss_dict['G_loss_adv'] = G_loss_adv
            loss_dict['GAN_Feat'] = GAN_Feat
            loss_dict['loss_VGG'] = loss_VGG
            loss_dict['loss_f'] = loss_f
            loss_dict['loss_semi'] = loss_Semi

            G_loss = G_loss_adv + GAN_Feat + loss_VGG + loss_f + loss_Semi

            G.zero_grad()
            G_loss.backward()
            G_optim.step()

            loss_log = ''
            for l in loss_dict:
                loss_log += l + ":" + str(round(loss_dict[l].item(), 4)) + '; '

            print(('epoch_{}_{}:' + loss_log).format(epoch,idx))
            epoch_loss_dir = os.path.join(args.result_loss_txt, 'epoch_' + str(epoch) + '.txt')
            with open(epoch_loss_dir, 'a+') as f:
                f.write(loss_log + '\n')

            if idx >= 0 and idx < 50:
                this_epoch = os.path.join(args.result_train_data, 'epoch_' + str(epoch))
                if not os.path.exists(this_epoch):
                    os.mkdir(this_epoch)

                imgs = torch.cat([supervisied_A, supervisied_B, supervisied_A2B, unsupervised_A, unsupervised_A2B], dim=3)


                utils.save_image(
                    imgs,
                    os.path.join(this_epoch, 'supervisied_' + str(count) + '.png'),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                count += 1
    torch.save({
        "g": G.state_dict()
    }, os.path.join(args.result_checkpoints, args.name + '.pt'))