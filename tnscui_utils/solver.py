import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision
import ttach
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.optim import lr_scheduler

from loss_func.dice_helpers import soft_cldice_loss
from loss_func.dice_loss import FocalTversky_loss
from loss_func.lovasz_losses import lovasz_hinge, binary_xloss
from tnscui_utils.TNSUCI_util import char_color, GradualWarmupScheduler
from utils.evaluation import *
from tqdm import tqdm

import segmentation_models as smp


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader, logging):

        self.threshold = 0.5
        # Make record file
        self.record_file = os.path.join(config.result_path, 'record.txt')
        f = open(self.record_file, 'w')
        f.close()

        self.Task_name = config.Task_name

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.train_list = config.train_list
        self.valid_list = config.valid_list
        self.test_list = config.test_list

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.img_size = config.image_size
        self.output_ch = config.output_ch
        self.augmentation_prob = config.augmentation_prob

        # loss
        self.criterion = lovasz_hinge
        self.criterion1 = binary_xloss

        self.criterion2 = SoftDiceLoss()
        self.criterion3 = FocalTversky_loss()
        self.criterion4 = soft_cldice_loss

        # Hyper-parameters
        self.lr = config.lr
        self.lr_low = config.lr_low
        if self.lr_low is None:
            self.lr_low = self.lr / 1e+6
            print("auto set minimun lr :", self.lr_low)

        # optimizer param
        self.beta1 = config.beta1  # for adam
        self.beta2 = config.beta2  # for adam

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.save_model_step = config.save_model_step
        self.val_step = config.val_step
        self.decay_step = config.decay_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        self.save_image = config.save_image
        self.save_detail_result = config.save_detail_result
        self.log_dir = config.log_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.DataParallel = config.DataParallel

        self.test_flag = config.test_flag

        # 设置学习率策略相关参数
        self.decay_ratio = config.decay_ratio
        self.lr_cos_epoch = config.lr_cos_epoch
        self.lr_warm_epoch = config.lr_warm_epoch
        self.lr_sch = None  # 初始化先设置为None
        self.lr_list = []  # 临时记录lr

        self.tta_mode = config.tta_mode

        self.logging = logging
        self.writer = SummaryWriter(log_dir=self.log_dir)
        # 执行个初始化函数
        self.my_init()

    def mylogging(self, *args):
        """Print & Record while training."""
        self.logging.info(*args)
        f = open(self.record_file, 'a')
        print(*args, file=f)
        f.close()

    def myprint(self, *args):
        """Print & Record while training."""
        print(*args)
        f = open(self.record_file, 'a')
        print(*args, file=f)
        f.close()

    def my_init(self):
        self.mylogging(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))
        self.print_data_msg()
        self.build_model()

    def print_data_msg(self):
        img_shape = (self.img_ch,self.img_size,self.img_size)
        info = f'''Starting training:
                image shape:     {img_shape}
                Epochs:          {self.num_epochs}
                Batch size:      {self.batch_size}
                Device:          {self.device}
                train images:    {len(self.train_list)}
                valid images:    {len(self.valid_list)}
                test images:     {len(self.test_list)}
                DataParallel     {self.DataParallel}
            '''
        self.mylogging(info)
        # self.mylogging("train:{} images".format(len(self.train_list)))
        # self.mylogging("valid:{} images".format(len(self.valid_list)))
        # self.mylogging(" test:{} images".format(len(self.test_list)))

    def build_model(self):
        # 在这里自己搭建自己的网络(网络结构)
        self.unet = smp.DeepLabV3Plus(encoder_name="efficientnet-b6",
                                      encoder_weights='imagenet',
                                      in_channels=self.img_ch, classes=1)

        # 优化器修改
        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])

        # lr schachle策略(要传入optimizer才可以)
        # 暂时有三种情况,(1)只用cosine decay,(2)只用warmup,(3)两者都用
        if self.lr_warm_epoch != 0 and self.lr_cos_epoch == 0:  # zhishiyong
            self.update_lr(self.lr_low)  # 使用warmup需要吧lr初始化为最小lr
            self.lr_sch = GradualWarmupScheduler(self.optimizer,
                                                 multiplier=self.lr / self.lr_low,
                                                 total_epoch=self.lr_warm_epoch,
                                                 after_scheduler=None)
            self.mylogging('use warmup lr sch')
        elif self.lr_warm_epoch == 0 and self.lr_cos_epoch != 0:
            self.lr_sch = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                         self.lr_cos_epoch,
                                                         eta_min=self.lr_low)
            self.mylogging('use cos lr sch')
        elif self.lr_warm_epoch != 0 and self.lr_cos_epoch != 0:
            self.update_lr(self.lr_low)  # 使用warmup需要吧lr初始化为最小lr
            scheduler_cos = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                           self.lr_cos_epoch,
                                                           eta_min=self.lr_low)
            self.lr_sch = GradualWarmupScheduler(self.optimizer,
                                                 multiplier=self.lr / self.lr_low,
                                                 total_epoch=self.lr_warm_epoch,
                                                 after_scheduler=scheduler_cos)
            self.mylogging('use warmup and cos lr sch')
        else:
            if self.lr_sch is None:
                self.mylogging('use linear decay')

        self.unet.to(self.device)
        if self.DataParallel:
            self.unet = torch.nn.DataParallel(self.unet)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        self.mylogging(model)
        self.mylogging(name)
        self.mylogging("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, lr):
        """Update the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def tensor2img(self, x):
        """Convert tensor to img (numpy)."""
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        """Train encoder, generator and discriminator."""
        # self.mylogging('-----------------------%s-----------------------------' % self.Task_name)
        unet_path = os.path.join(self.model_path, 'best_unet_score.pkl')

        # 断店继训练,看看是否有上一训练时候保存的最优模型
        # self.logging.info(unet_path + str(os.path.isfile(unet_path)))
        if os.path.isfile(unet_path):  # False:
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            self.mylogging(char_color('Successfully Loaded from %s' % (unet_path),word=33))

        # Train for Encoder
        best_unet_score = 0.
        Iter = 0
        train_len = len(self.train_loader)
        valid_record = np.zeros((1, 8))  # [epoch, Iter, acc, SE, SP, PC, Dice, IOU]
        test_record = np.zeros((1, 8))  # [epoch, Iter, acc, SE, SP, PC, Dice, IOU]

        self.mylogging(char_color('Training...',word=35))
        for epoch in range(self.num_epochs):
            tic = datetime.datetime.now()
            self.unet.train(True)
            epoch_loss = 0
            length = 0
            with tqdm(total=train_len, desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch') as pbar:
                for batch in self.train_loader:
                    images, GT = batch[1], batch[2]
                    images = images.to(self.device)
                    GT = GT.to(self.device)

                    # SR : Segmentation Result
                    SR = self.unet(images)

                    SR_probs = F.sigmoid(SR)
                    SR_flat = SR_probs.view(SR_probs.size(0), -1)

                    GT_flat = GT.view(GT.size(0), -1)

                    SR_logits_sq = torch.squeeze(SR)
                    GT_sqz = torch.squeeze(GT)
                    # print(SR_logits_sq.shape)
                    # print(GT_sqz.shape)

                    loss_softdice = self.criterion2(SR_flat, GT_flat)
                    loss_lovz = self.criterion(SR_logits_sq, GT_sqz)
                    loss_bi_BCE = self.criterion1(SR_logits_sq, GT_sqz)
                    # loss_clsoftdice = torch.mean(self.criterion4(SR_probs, GT))

                    loss = 0.0 * loss_lovz + 1.0 * loss_softdice + 0.0 * loss_bi_BCE
                    # loss = 1.0*loss_softdice
                    epoch_loss += float(loss)

                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.writer.add_scalars('Loss', {'loss': loss.item()}, Iter)
                    pbar.set_postfix(**{'loss(batch)': loss.item()})

                    if self.save_image and (Iter % 20 == 0):
                        images_all = torch.cat((images, SR, GT), 0)
                        torchvision.utils.save_image(images_all.data.cpu(),
                                                     os.path.join(self.result_path, 'image',
                                                                  'Train_%d_image.png' % Iter),
                                                     nrow=self.batch_size)
                        self.writer.add_images('images', images, Iter)
                        self.writer.add_images('mask/true', GT, Iter)
                        self.writer.add_images('mask/pred', SR, Iter)
                    pbar.update(1)
                    length += 1
                    Iter += 1

            # 计时结束
            toc = datetime.datetime.now()
            h, remainder = divmod((toc - tic).seconds, 3600)
            m, s = divmod(remainder, 60)
            self.mylogging(f'per epoch training cost Time {h} h:{m} m:{s} s')

            tic = datetime.datetime.now()

            epoch_loss = epoch_loss / length
            self.mylogging('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, self.num_epochs, epoch_loss))

            # 记录下lr到log里(并且记录到图片里)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_list.append(current_lr)
            self.writer.add_scalar('lr', current_lr, epoch)
            # 保存lr为png
            # figg = plt.figure()
            # plt.plot(self.lr_list)
            # figg.savefig(os.path.join(self.result_path, 'lr.PNG'))
            # plt.close()

            # figg, axis = plt.subplots()
            # plt.plot(self.lr_list)
            # axis.set_yscale("log")
            # figg.savefig(os.path.join(self.result_path, 'lr_log.PNG'))
            # plt.close()

            # 学习率策略部分 =========================
            # lr scha way 1:
            if self.lr_sch is not None:
                if (epoch + 1) <= (self.lr_cos_epoch + self.lr_warm_epoch):
                    self.lr_sch.step()

            # lr scha way 2: Decay learning rate(如果使用方式1,则不使用此方式)
            if self.lr_sch is None:
                if ((epoch + 1) >= self.num_epochs_decay) and (
                        (epoch + 1 - self.num_epochs_decay) % self.decay_step == 0):
                    if current_lr >= self.lr_low:
                        self.lr = current_lr * self.decay_ratio
                        # self.lr /= 100.0
                        self.update_lr(self.lr)
                        self.mylogging('Decay learning rate to lr: {}.'.format(self.lr))

            # Validation & Test
            if (epoch + 1) % self.val_step == 0:
                # Validation #
                if self.tta_mode:
                    self.mylogging(char_color('Testing with TTA',word=34))
                    acc, SE, SP, PC, DC, IOU = self.test(mode='valid', tta=True)
                else:
                    self.mylogging(char_color('Testing',word=34))
                    acc, SE, SP, PC, DC, IOU = self.test(mode='valid', tta=False)
                valid_record = np.vstack((valid_record, np.array([epoch + 1, Iter, acc, SE, SP, PC, DC, IOU])))
                unet_score = IOU  # TODO
                self.writer.add_scalars('Valid', {'Dice': DC, 'IOU': IOU}, epoch)
                self.mylogging('[Validation] Dice: %.4f, IOU: %.4f' % (DC, IOU))

                # Save Best U-Net model
                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    self.mylogging(
                        char_color('Best model in epoch %d, score(IOU) : %.4f' % (best_epoch + 1, best_unet_score),
                                   word=33))
                    torch.save(best_unet, unet_path)

                #  Test
                if self.test_flag:
                    if self.tta_mode:
                        acc, SE, SP, PC, DC, IOU = self.test(mode='test', tta=True)
                    else:
                        acc, SE, SP, PC, DC, IOU = self.test(mode='test', tta=False)
                    test_record = np.vstack(((test_record, np.array([epoch + 1, Iter, acc, SE, SP, PC, DC, IOU]))))
                    self.writer.add_scalars('Test', {'Dice': DC, 'IOU': IOU, 'Acc': acc}, epoch)
                    self.mylogging('[Testing]    Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, Dice: %.4f, IOU: %.4f' % (
                        acc, SE, SP, PC, DC, IOU))

                # save_record_in_xlsx
                if True:
                    excel_save_path = os.path.join(self.result_path, 'record.xlsx')
                    record = pd.ExcelWriter(excel_save_path)
                    detail_result1 = pd.DataFrame(valid_record)
                    detail_result1.to_excel(record, 'valid', float_format='%.5f')
                    if self.test_flag:
                        detail_result2 = pd.DataFrame(test_record)
                        detail_result2.to_excel(record, 'test', float_format='%.5f')
                    record.save()
                    record.close()
                toc = datetime.datetime.now()
                h, remainder = divmod((toc - tic).seconds, 3600)
                m, s = divmod(remainder, 60)
                time_str = "Testing & vlidation cost Time %02d h:%02d m:%02d s" % (h, m, s)
                self.mylogging(time_str)

            # save model
            if (epoch + 1) % self.save_model_step == 0:
                save_unet = self.unet.state_dict()
                if not os.path.exists(self.model_path):
                    os.mkdir(self.model_path)
                torch.save(save_unet, os.path.join(self.model_path, 'epoch%d.pkl' % (epoch + 1)))

        self.writer.close()
        self.mylogging('Finished!')
        self.mylogging(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))

    def test(self, mode='test', unet_path=None, tta=True):
        """Test model & Calculate performances."""
        if not unet_path is None:
            if os.path.isfile(unet_path):
                self.unet.load_state_dict(torch.load(unet_path))
                self.mylogging(char_color('Successfully Loaded from %s' % (unet_path),word=36))

        self.unet.train(False)
        self.unet.eval()

        if mode == 'train':
            data_lodear = self.train_loader
        elif mode == 'test':
            data_lodear = self.test_loader
        elif mode == 'valid':
            data_lodear = self.valid_loader

        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        DC = 0.  # Dice Coefficient
        IOU = 0.  # IOU
        length = 0

        test_len = len(data_lodear)
        # model pre for each image
        detail_result = []  # detail_result = [id, acc, SE, SP, PC, dsc, IOU]
        i = 0
        with torch.no_grad():
            with tqdm(total=test_len, desc=f'Test', unit='batch') as pbar:
                for batch in data_lodear:
                    (image_paths, images, GT) = batch
                    images_path = list(image_paths)
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    if tta:
                        tta_trans = ttach.Compose([
                            ttach.VerticalFlip(),
                            ttach.HorizontalFlip(),
                            ttach.Rotate90(angles=[0, 180])
                        ])
                        tta_model = ttach.SegmentationTTAWrapper(self.unet, tta_trans)
                        SR = tta_model(images)
                    else:
                        SR = self.unet(images)

                    SR = F.sigmoid(SR)
                    if self.save_image:
                        images_all = torch.cat((images, SR, GT), 0)
                        torchvision.utils.save_image(images_all.data.cpu(),
                                                     os.path.join(self.result_path, 'image',
                                                                  '%s_%d_image.png' % (mode, i)),
                                                     nrow=self.batch_size)

                    SR = SR.data.cpu().numpy()
                    GT = GT.data.cpu().numpy()

                    for ii in range(SR.shape[0]):
                        SR_tmp = SR[ii, :].reshape(-1)
                        GT_tmp = GT[ii, :].reshape(-1)
                        tmp_index = images_path[ii].split('/')[-1]
                        tmp_index = int(tmp_index.split('.')[0][:])

                        # SR_tmp = torch.from_numpy(SR_tmp).to(self.device)
                        # GT_tmp = torch.from_numpy(GT_tmp).to(self.device)

                        result_tmp = np.array([tmp_index,
                                               get_accuracy(SR_tmp, GT_tmp, self.threshold),
                                               get_sensitivity(SR_tmp, GT_tmp, self.threshold),
                                               get_specificity(SR_tmp, GT_tmp, self.threshold),
                                               get_precision(SR_tmp, GT_tmp, self.threshold),
                                               get_DC(SR_tmp, GT_tmp, self.threshold),
                                               get_IOU(SR_tmp, GT_tmp, self.threshold)])

                        acc += result_tmp[1]
                        SE += result_tmp[2]
                        SP += result_tmp[3]
                        PC += result_tmp[4]
                        DC += result_tmp[5]
                        IOU += result_tmp[6]
                        detail_result.append(result_tmp)

                        length += 1
                    i += 1
                    pbar.set_postfix(**{'acc': acc / length, 'DC': DC / length, 'IOU': IOU / length})
                    pbar.update(1)

        accuracy = acc / length
        sensitivity = SE / length
        specificity = SP / length
        precision = PC / length
        disc = DC / length
        iou = IOU / length
        detail_result = np.array(detail_result)

        if self.save_detail_result:  # detail_result = [id, acc, SE, SP, PC, dsc, IOU]
            excel_save_path = os.path.join(self.result_path, mode + '_pre_detial_result.xlsx')
            writer = pd.ExcelWriter(excel_save_path)
            detail_result = pd.DataFrame(detail_result)
            detail_result.to_excel(writer, mode, float_format='%.5f')
            writer.save()
            writer.close()

        return accuracy, sensitivity, specificity, precision, disc, iou
