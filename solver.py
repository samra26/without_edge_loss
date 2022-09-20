import torch
from torch.nn import functional as F
from conformer import build_model
import numpy as np
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
import torch.nn as nn
import argparse
import os.path as osp
import os
size_coarse = (10, 10)



class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        #self.build_model()
        self.net = build_model(self.config.network, self.config.arch)
        #self.net.eval()
        if config.mode == 'test':
            print('Loading pre-trained model for testing from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model, map_location=torch.device('cpu')))
        if config.mode == 'train':
            if self.config.load == '':
                print("Loading pre-trained imagenet weights for fine tuning")
                self.net.JLModule.load_pretrained_model(self.config.pretrained_model
                                                        if isinstance(self.config.pretrained_model, str)
                                                        else self.config.pretrained_model[self.config.network])
                # load pretrained backbone
            else:
                print('Loading pretrained model to resume training')
                self.net.load_state_dict(torch.load(self.config.load))  # load pretrained model
        
        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'Conformer based SOD Structure')

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params_t = 0
        num_params=0
        for p in model.parameters():
            if p.requires_grad:
                num_params_t += p.numel()
            else:
                num_params += p.numel()
        print(name)
        print(model)
        print("The number of trainable parameters: {}".format(num_params_t))
        print("The number of parameters: {}".format(num_params))

    # build the network
    '''def build_model(self):
        self.net = build_model(self.config.network, self.config.arch)

        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'JL-DCF Structure')'''

    def test(self):
        print('Testing...')
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']), \
                                           data_batch['depth']
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)

                #input = torch.cat((images, depth), dim=0)
                preds,sal_low,sal_med,sal_high,coarse_sal_rgb,coarse_sal_depth,Att,e_rgbd0,e_rgbd1,e_rgbd2 = self.net(images,depth)
                #print(e_rgbd01.shape)
                preds = F.interpolate(preds, tuple(im_size), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()
                #print(pred.shape)
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_folder, name[:-4] + '_convtran.png')
                cv2.imwrite(filename, multi_fuse)
                coarse_sal_rgb= F.interpolate(coarse_sal_rgb, tuple(im_size), mode='bilinear', align_corners=True)
                coarse_sal_rgbs = np.squeeze(torch.sigmoid(coarse_sal_rgb)).cpu().data.numpy()
                #print(pred.shape)
                coarse_sal_rgbs = (coarse_sal_rgbs - coarse_sal_rgbs.min()) / (coarse_sal_rgbs.max() - coarse_sal_rgbs.min() + 1e-8)
                multi_fuse_coarse_sal_rgb = 255 * coarse_sal_rgbs
                filename_r = os.path.join(self.config.test_folder, name[:-4] + '_coarse_sal_rgb.png')
                cv2.imwrite(filename_r, multi_fuse_coarse_sal_rgb)
                coarse_sal_depth= F.interpolate(coarse_sal_depth, tuple(im_size), mode='bilinear', align_corners=True)
                coarse_sal_ds = np.squeeze(torch.sigmoid(coarse_sal_depth)).cpu().data.numpy()
                #print(pred.shape)
                coarse_sal_ds = (coarse_sal_ds - coarse_sal_ds.min()) / (coarse_sal_ds.max() - coarse_sal_ds.min() + 1e-8)
                multi_fuse_coarse_sal_ds = 255 * coarse_sal_ds
                filename_d = os.path.join(self.config.test_folder, name[:-4] + '_coarse_sal_d.png')
                cv2.imwrite(filename_d, multi_fuse_coarse_sal_ds)
                e_rgbd2= F.interpolate(e_rgbd2, tuple(im_size), mode='bilinear', align_corners=True)
                e_rgbd2 = np.squeeze(torch.sigmoid(e_rgbd2)).cpu().data.numpy()
                #print(pred.shape)
                e_rgbd2 = (e_rgbd2 - e_rgbd2.min()) / (e_rgbd2.max() - e_rgbd2.min() + 1e-8)
                multi_fuse_e_rgbd2 = 255 * e_rgbd2
                filename_re = os.path.join(self.config.test_folder, name[:-4] + '_edge2.png')
                cv2.imwrite(filename_re, multi_fuse_e_rgbd2)
                '''#e_rgbd01 = F.interpolate(e_rgbd01, tuple(im_size), mode='bilinear', align_corners=True)
                e_rgbd01 = np.squeeze(torch.sigmoid(Att[10])).cpu().data.numpy()
                print(e_rgbd01.shape)
                #e_rgbd01 = (e_rgbd01-e_rgbd01.min()) / (e_rgbd01.max() - e_rgbd01.min() + 1e-8)
                #e_rgbd01 = 255 * e_rgbd01
                filename = os.path.join(self.config.test_folder, name[:-4] + '_edge.png')
                a=cv2.imwrite(filename, e_rgbd01)
                print(a)
                #e_rgbd11 = F.interpolate(e_rgbd11, tuple(im_size), mode='bilinear', align_corners=True)
                e_rgbd11 = np.squeeze(torch.sigmoid(Att[11])).cpu().data.numpy()
                print(e_rgbd11.shape)
                e_rgbd11 = (e_rgbd11-e_rgbd11.min()) / (e_rgbd11.max() - e_rgbd11.min() + 1e-8)
                e_rgbd11 = 255 * e_rgbd11
                filename = os.path.join(self.config.test_folder, name[:-5] + '_edge.png')
                cv2.imwrite(filename, e_rgbd11)

                #e_rgbd21 = F.interpolate(e_rgbd21, tuple(im_size), mode='bilinear', align_corners=True)
                e_rgbd21 = np.squeeze(torch.sigmoid(Att[9])).cpu().data.numpy()

                e_rgbd21 = (e_rgbd21-e_rgbd21.min()) / (e_rgbd21.max() - e_rgbd21.min() + 1e-8)
                e_rgbd21 = 255 * e_rgbd21
                filename = os.path.join(self.config.test_folder, name[:-6] + '_edge.png')
                cv2.imwrite(filename, e_rgbd21)'''
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')
    
  
    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        
        loss_vals=  []
        
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            r_sal_loss_item=0
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_depth, sal_label, sal_edge = data_batch['sal_image'], data_batch['sal_depth'], data_batch[
                    'sal_label'], data_batch['sal_edge']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image, sal_depth, sal_label, sal_edge= sal_image.to(device), sal_depth.to(device), sal_label.to(device),sal_edge.to(device)

               
                self.optimizer.zero_grad()
                sal_label_coarse = F.interpolate(sal_label, size_coarse, mode='bilinear', align_corners=True)
                
                sal_final,sal_low,sal_med,sal_high,coarse_sal_rgb,coarse_sal_depth,Att,sal_edge_rgbd0,sal_edge_rgbd1,sal_edge_rgbd2 = self.net(sal_image,sal_depth)
                
                sal_loss_coarse_rgb =  F.binary_cross_entropy_with_logits(coarse_sal_rgb, sal_label_coarse, reduction='sum')
                sal_loss_coarse_depth =  F.binary_cross_entropy_with_logits(coarse_sal_depth, sal_label_coarse, reduction='sum')
                sal_final_loss =  F.binary_cross_entropy_with_logits(sal_final, sal_label, reduction='sum')
              
                
                sal_loss_fuse = sal_final_loss+sal_loss_coarse_rgb+sal_loss_coarse_depth
                sal_loss = sal_loss_fuse/ (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data
                r_sal_loss_item+=sal_loss.item() * sal_image.size(0)
                sal_loss.backward()
                self.optimizer.step()

                if (i + 1) % (self.show_every // self.config.batch_size) == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %0.4f  ||sal_final:%0.4f||  r:%0.4f||d:%0.4f' % (
                        epoch, self.config.epoch, i + 1, iter_num, r_sal_loss,sal_final_loss,sal_loss_coarse_rgb,sal_loss_coarse_depth ))
                    # print('Learning rate: ' + str(self.lr))
                    writer.add_scalar('training loss', r_sal_loss / (self.show_every / self.iter_size),
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar('sal_loss_coarse_rgb training loss', sal_loss_coarse_rgb.data,
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar('sal_loss_coarse_depth training loss', sal_loss_coarse_depth.data,
                                      epoch * len(self.train_loader.dataset) + i)
                  

                    r_sal_loss = 0
                    res = coarse_sal_depth[0].clone()
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    writer.add_image('coarse_sal_depth', torch.tensor(res), i, dataformats='HW')
                    grid_image = make_grid(sal_label_coarse[0].clone().cpu().data, 1, normalize=True)

                    res = coarse_sal_rgb[0].clone()
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    writer.add_image('coarse_sal_rgb', torch.tensor(res), i, dataformats='HW')
                    grid_image = make_grid(sal_label_coarse[0].clone().cpu().data, 1, normalize=True)
                    
                    fsal = sal_final[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_final', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)

                    fsal = sal_low[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_low', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)
                    fsal = sal_high[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_high', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)
                    fsal = sal_med[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_med', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)
                   


            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
            train_loss=r_sal_loss_item/len(self.train_loader.dataset)
            loss_vals.append(train_loss)
            
            print('Epoch:[%2d/%2d] | Train Loss : %.3f' % (epoch, self.config.epoch,train_loss))
            
        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)
        

