import os
import logging
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import ttach

from torch import nn, optim
from torchvision import transforms
from torchvision import datasets
from torch.backends import cudnn
from loader.data_loader_gan import get_loader, get_loader_difficult
from utils.evaluation import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from tnscui_utils.TNSUCI_util import *
from utils.img_utils import preprocess, largestConnectComponent
from skimage.transform import resize

import segmentation_models as smp


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels,
                           kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, conv_dim=32):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        # 512 x 512 x 1
        self.cv1 = conv(1, self.conv_dim, 3, batch_norm=False)
        # 256 x 256 x 32
        self.cv2 = conv(self.conv_dim, self.conv_dim * 2, 3, batch_norm=True)
        # 128 x 128 x 64
        self.cv3 = conv(self.conv_dim * 2, self.conv_dim * 4, 3, batch_norm=True)
        # 64 x 64 x 128
        self.cv4 = conv(self.conv_dim * 4, self.conv_dim * 8, 3, batch_norm=True)
        # 32 x 32 x 256
        self.cv5 = conv(self.conv_dim * 8, self.conv_dim * 16, 3, batch_norm=True)
        # 16 x 16 x 512
        self.fc1 = nn.Linear(self.conv_dim * 16 * 16 * 16, 1)

    def forward(self, x):
        x = F.leaky_relu(self.cv1(x), 0.2)
        x = F.leaky_relu(self.cv2(x), 0.2)
        x = F.leaky_relu(self.cv3(x), 0.2)
        x = F.leaky_relu(self.cv4(x), 0.2)
        x = F.leaky_relu(self.cv5(x), 0.2)
        x = x.view(-1, self.conv_dim * 16 * 16 * 16)
        x = self.fc1(x)
        return x


class StageOneTrainer(object):

    def __init__(self, args, train_loader, val_loader, test_loader, logging):

        self.c1_size = args.c1_size
        self.c2_size = args.c2_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # data loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Net
        self.GNet = smp.DeepLabV3Plus(encoder_name="efficientnet-b6",
                                      encoder_weights='imagenet',
                                      in_channels=args.img_ch, classes=1).to(self.device)
        self.DNet = Discriminator(conv_dim=32).to(self.device)

        # optimizer
        self.G_optimizer = optim.Adam(self.GNet.parameters(), args.lr, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.DNet.parameters(), args.lr, betas=(args.beta1, args.beta2))
        self.G_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.G_optimizer, 'min', patience=2)
        self.D_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.D_optimizer, 'min', patience=2)

        # loss
        self.seg_loss = SoftDiceLoss()
        self.args = args

        self.logging = logging

        self.writer = SummaryWriter(log_dir=args.log_dir)
        if 'train' in args.action:
            inputs1 = next(iter(train_loader))[1]
            self.writer.add_graph(self.GNet, inputs1.to(self.device, dtype=torch.float32))
            # inputs2 = next(iter(train_loader))[3]
            # self.writer.add_graph(self.DNet, inputs2.to(self.device, dtype=torch.float32))

        if self.args.DataParallel:
            self.GNet = torch.nn.DataParallel(self.GNet)
            self.DNet = torch.nn.DataParallel(self.DNet)

    def train(self):
        if os.path.isfile(os.path.join(self.args.save_path, 'best_model_G.pth')):
            self.GNet.load_state_dict(torch.load(os.path.join(self.args.save_path, 'best_model_G.pth')))
            self.logging.info(
                char_color(f'load weights:{os.path.join(self.args.save_path, "best_model_G.pth")} finish!', word=36))
        if os.path.isfile(os.path.join(self.args.save_path, 'best_model_D.pth')):
            self.DNet.load_state_dict(torch.load(os.path.join(self.args.save_path, 'best_model_D.pth')))
            self.logging.info(
                char_color(f'load weights:{os.path.join(self.args.save_path, "best_model_G.pth")} finish!', word=36))

        n_train = len(self.train_loader.dataset)
        step = 0
        best_DsC = 0.
        best_IoU = 0.
        for epoch in range(self.args.epochs):

            epoch_G_loss = 0
            epoch_D_loss = 0
            # training
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{self.args.epochs}', unit='img') as pbar:
                for batch in self.train_loader:
                    images, masks, small_images = batch[1], batch[2], batch[3]
                    images = images.to(device=self.device, dtype=torch.float32)
                    masks = masks.to(device=self.device, dtype=torch.float32)
                    small_images = small_images.to(device=self.device, dtype=torch.float32)

                    self.DNet.train()
                    self.GNet.eval()
                    # train Discriminator
                    self.D_optimizer.zero_grad()
                    D_real = self.DNet(small_images)
                    real_loss = self.real_loss(D_real)

                    pred_masks = self.GNet(images)
                    pred_masks = F.sigmoid(pred_masks)
                    fake_small_images = self.cut_image(masks, pred_masks)
                    D_fake = self.DNet(fake_small_images)
                    fake_loss = self.fake_loss(D_fake)

                    D_loss = real_loss + fake_loss
                    D_loss.backward()
                    self.D_optimizer.step()
                    epoch_D_loss += D_loss.item()

                    self.GNet.train()
                    self.DNet.eval()
                    # train Generator
                    self.G_optimizer.zero_grad()
                    pred_masks = self.GNet(images)
                    pred_masks = F.sigmoid(pred_masks)
                    pred_flat = pred_masks.view(pred_masks.size(0), -1)
                    masks_flat = masks.view(masks.size(0), -1)
                    seg_loss = self.seg_loss(pred_flat, masks_flat)

                    fake_small_images = self.cut_image(masks, pred_masks)
                    D_fake = self.DNet(fake_small_images)
                    loss = self.real_loss(D_fake, False)
                    G_loss = seg_loss + 0.1 * loss
                    G_loss.backward()
                    self.G_optimizer.step()
                    epoch_G_loss += G_loss.item()

                    self.writer.add_images('cut/true', small_images, step)
                    self.writer.add_images('cut/fake', fake_small_images, step)
                    self.writer.add_scalar('Loss_G/seg', seg_loss.item(), step)
                    self.writer.add_scalar('Loss_G/adv', loss.item(), step)
                    self.writer.add_scalar('Loss_G/total', G_loss.item(), step)
                    self.writer.add_scalar('Loss_D', D_loss.item(), step)
                    pbar.set_postfix(**{'loss(G)': G_loss.item(), 'loss_seg': seg_loss.item(), 'loss_adv': loss.item(),
                                        'loss(D)': D_loss.item()})
                    pbar.update(images.shape[0])
                    step = step + 1

            self.G_scheduler.step(epoch_G_loss)
            self.D_scheduler.step(epoch_D_loss)
            # eval
            if (epoch + 1) % self.args.val_epoch == 0:
                DsC, IoU = self.test(mode='val')
                if DsC > best_DsC:
                    best_DsC = DsC
                    if self.args.save_path:
                        if not os.path.exists(self.args.save_path):
                            os.makedirs(self.args.save_path)
                        torch.save(self.GNet.state_dict(), f'{self.args.save_path}/best_model_G.pth')
                        torch.save(self.DNet.state_dict(), f'{self.args.save_path}/best_model_D.pth')
                        self.logging.info(char_color(f'best model saved !', word=33))

                self.logging.info(f'DsC: {DsC},IoU：{IoU}')
                self.writer.add_scalars('Valid', {'DsC': DsC, 'IOU': IoU}, step)
                self.writer.add_scalar('learning_rate', self.G_optimizer.param_groups[0]['lr'], step)
                self.writer.add_images('images', images, step)
                self.writer.add_images('masks/true', masks, step)
                self.writer.add_images('masks/pred', pred_masks, step)

            if (epoch + 1) % self.args.save_model_epoch == 0:
                if self.args.save_path:
                    if not os.path.exists(self.args.save_path):
                        os.makedirs(self.args.save_path)
                    G_model_name = f'{self.args.Task_name}_G_'
                    D_model_name = f'{self.args.Task_name}_D_'
                    torch.save(self.GNet.state_dict(),
                               f'{self.args.save_path}/{G_model_name}{epoch + 1}.pth')
                    torch.save(self.DNet.state_dict(),
                               f'{self.args.save_path}/{D_model_name}{epoch + 1}.pth')
                    self.logging.info(char_color(f'Checkpoint {epoch + 1} saved !'))
        self.writer.close()

    def test(self, mode='test', model_path=None, aug=False):
        if not model_path is None:
            if os.path.isfile(model_path):
                self.GNet.load_state_dict(torch.load(model_path))
                self.logging.info(char_color('Successfully Loaded from %s' % (model_path), word=36))

        self.GNet.train(False)
        self.GNet.eval()
        if mode == 'test':
            data_lodear = self.test_loader
        elif mode == 'val':
            data_lodear = self.val_loader

        test_len = len(data_lodear)
        IoU = 0.
        DsC = 0.
        length = 0
        i = 0
        IsU_list = []
        DsC_list = []
        with torch.no_grad():
            with tqdm(total=test_len, desc=f'{mode}', unit='batch') as pbar:
                for batch in data_lodear:
                    images, masks = batch[1], batch[2]
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    if aug:
                        tta_trans = ttach.Compose([
                            ttach.VerticalFlip(),
                            ttach.HorizontalFlip(),
                            ttach.Rotate90(angles=[0, 180])
                        ])
                        tta_model = ttach.SegmentationTTAWrapper(self.GNet, tta_trans)
                        pred = tta_model(images)
                    else:
                        pred = self.GNet(images)
                    # pred = torch.sigmoid(pred)
                    pred = pred.data.cpu().numpy()
                    masks = masks.data.cpu().numpy()
                    for j in range(pred.shape[0]):
                        pre_tmp = pred[j, :].reshape(-1)
                        GT_tmp = masks[j, :].reshape(-1)
                        DsC += get_DC(pre_tmp, GT_tmp, self.args.threshold)
                        IoU += get_IOU(pre_tmp, GT_tmp, self.args.threshold)
                        IsU_list.append(IoU)
                        DsC_list.append(DsC)
                        length += 1
                    i += 1
                    pbar.set_postfix(**{'DsC': DsC / length, 'IoU': IoU / length})
                    pbar.update(1)
        if mode == 'test':
            print('Before screen:')
            print(char_color(f'Final IoU:{np.mean(IsU_list)}'))
            print(char_color(f'Final Dice:{np.mean(DsC_list)}'))
            IoU_list = [x for x in IsU_list if x > 0.4]
            DsC_list = [x for x in DsC_list if x > 0.4]
            print('After screen:')
            print(char_color(f'Final IoU:{np.mean(IoU_list)}', word=34))
            print(char_color(f'Final Dice:{np.mean(DsC_list)}', word=34))
        return (DsC / length), (IoU / length)

    def real_loss(self, D_out, smooth=False):
        batch_size = D_out.size(0)
        if smooth:
            labels = torch.ones(batch_size) * 0.9
        else:
            labels = torch.ones(batch_size)

        labels = labels.to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(D_out.squeeze(), labels)
        return loss

    def fake_loss(self, D_out):
        batch_size = D_out.size(0)
        labels = torch.zeros(batch_size)
        labels = labels.to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(D_out.squeeze(), labels)
        return loss

    def cut_image(self, images, masks):
        images_array = (torch.squeeze(images)).data.cpu().numpy()
        masks_array = (torch.squeeze(masks)).data.cpu().numpy()
        masks_array = (masks_array > 0.5)
        masks_array = masks_array.astype(np.float32)
        cut_images = []
        for i in range(masks_array.shape[0]):
            # mask_array_biggest = largestConnectComponent(masks_array[i].astype(np.int))
            mask_array_biggest = masks_array[i].astype(np.int)
            dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max = preprocess(mask_array_biggest, self.c1_size)
            img_array_roi = images_array[i][dim1_cut_min:dim1_cut_max, dim2_cut_min:dim2_cut_max]
            cut_image = resize(img_array_roi, (self.c2_size, self.c2_size), order=3)
            cut_images.append(cut_image[np.newaxis, :])
        return torch.tensor(np.array(cut_images)).to(self.device)


def get_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(message)s'
    )
    return logging


def get_args():
    parser = argparse.ArgumentParser()

    dataset = 'our_large'
    stage = 'stage2'
    # model hyper-parameters
    parser.add_argument('--c1_size', type=int, default=256)  # 网络输入img的size, 即输入会被强制resize到这个大小
    parser.add_argument('--c2_size', type=int, default=512)

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--num_epochs_decay', type=int, default=60)  # decay开始的最小epoch数
    parser.add_argument('--decay_ratio', type=float, default=0.01)  # 0~1,每次decay到1*ratio
    parser.add_argument('--decay_step', type=int, default=60)  # epoch

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_test', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=3)

    # 设置学习率
    parser.add_argument('--lr', type=float, default=1e-3)  # 初始or最大学习率(单用lovz且多gpu的时候,lr貌似要大一些才可收敛)
    parser.add_argument('--lr_low', type=float, default=1e-12)  # 最小学习率,设置为None,则为最大学习率的1e+6分之一(不可设置为0)

    parser.add_argument('--lr_warm_epoch', type=int, default=5)  # warmup的epoch数,一般就是5~20,为0或False则不使用
    parser.add_argument('--lr_cos_epoch', type=int, default=350)  # cos退火的epoch数,一般就是总epoch数-warmup的数,为0或False则代表不使用

    # optimizer param
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

    parser.add_argument('--augmentation_prob', type=float, default=1.0)  # 扩增几率

    parser.add_argument('--save_model_epoch', type=int, default=20)
    parser.add_argument('--val_epoch', type=int, default=1)

    # misc
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--tta_mode', type=bool, default=True)  # 是否在训练过程中的validation使用tta
    parser.add_argument('--Task_name', type=str, default=f'{dataset}_GAN_B6', help='DIR name,Task name')
    parser.add_argument('--cuda_ids', type=str, default='0,1')
    parser.add_argument('--DataParallel', type=bool, default=True)  ##
    # data-parameters
    parser.add_argument('--filepath_img', type=str, default=f'dataset/{dataset}/preprocessed/stage1/p_image')
    parser.add_argument('--filepath_mask', type=str, default=f'dataset/{dataset}/preprocessed/stage1/p_mask')
    parser.add_argument('--small_filepath_img', type=str, default=f'dataset/{dataset}/preprocessed/stage2/p_mask')
    parser.add_argument('--csv_file', type=str, default=f'dataset/{dataset}/preprocessed/train.csv')
    parser.add_argument('--fold_K', type=int, default=5, help='folds number after divided')
    parser.add_argument('--fold_idx', type=int, default=1)

    # result&save
    parser.add_argument('--result_path', type=str, default=f'./result')
    parser.add_argument('--save_path', type=str, default=os.path.join(parser.parse_args().result_path, 'models'),
                        help='the path of model weight file')
    parser.add_argument('--save_detail_result', type=bool, default=True)
    parser.add_argument('--save_image', type=bool, default=True)  # 训练过程中观察图像和结果

    # more param
    parser.add_argument('--test_flag', type=bool, default=False)  # 训练过程中是否测试,不测试会节省很多时间
    parser.add_argument('--validate_flag', type=bool, default=False)  # 是否有验证集
    parser.add_argument('--aug_type', type=str, default='difficult',
                        help='difficult or easy')  # 训练过程中扩增代码,分为dasheng,shaonan

    parser.add_argument('--action', type=str, help='train/test/train&test', default='train&test')
    parser.add_argument('--threshold', type=float, default=0.4)

    return parser.parse_args()


def main(config):
    logging = get_logging()
    cudnn.benchmark = True
    config.result_path = os.path.join(config.result_path, config.Task_name)
    logging.info(config.result_path)
    config.save_path = os.path.join(config.result_path, 'models')
    config.log_dir = os.path.join(config.result_path, 'logs')
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
        os.makedirs(os.path.join(config.result_path, 'image'))

    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_ids
    if not config.DataParallel:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    f = open(os.path.join(config.result_path, 'config.txt'), 'w')
    for key in config.__dict__:
        # logging.info(f'''{key}:{config.__getattribute__(key)}''')
        print('%s: %s' % (key, config.__getattribute__(key)), file=f)
    f.close()

    if config.validate_flag:
        train, valid, test = get_fold_filelist(config.csv_file, K=config.fold_K, fold=config.fold_idx, validation=True)

    else:
        train, test = get_fold_filelist(config.csv_file, K=config.fold_K, fold=config.fold_idx)
    """
    if u want to use fixed folder as img & mask folder, u can use following code 

    train_list = get_filelist_frompath(train_img_folder,'PNG') 
    train_list_GT = [train_mask_folder+sep+i.split(sep)[-1] for i in train_list]
    test_list = get_filelist_frompath(test_img_folder,'PNG') 
    test_list_GT = [test_mask_folder+sep+i.split(sep)[-1] for i in test_list]

    """

    train_list = [config.filepath_img + sep + i[0] for i in train]
    small_train_list = [config.small_filepath_img + sep + i[0] for i in train]
    train_list_GT = [config.filepath_mask + sep + i[0] for i in train]

    test_list = [config.filepath_img + sep + i[0] for i in test]
    test_list_GT = [config.filepath_mask + sep + i[0] for i in test]

    if config.validate_flag:
        valid_list = [config.filepath_img + sep + i[0] for i in valid]
        valid_list_GT = [config.filepath_mask + sep + i[0] for i in valid]
    else:
        # just copy test as validation,
        # also u can get the real valid_list use the func 'get_fold_filelist' by setting the param 'validation' as True
        valid_list = test_list
        valid_list_GT = test_list_GT

    config.train_list = train_list
    config.test_list = test_list
    config.valid_list = valid_list

    if config.aug_type == 'easy':
        logging.info('augmentation with easy level')
        train_loader = get_loader(seg_list=small_train_list,
                                  GT_list=train_list_GT,
                                  image_list=train_list,
                                  image_size=config.c1_size,
                                  batch_size=config.batch_size,
                                  load_preseg=True,
                                  num_workers=config.num_workers,
                                  mode='train',
                                  augmentation_prob=config.augmentation_prob, )
    elif config.aug_type == 'difficult':
        logging.info('augmentation with difficult level')
        train_loader = get_loader_difficult(seg_list=small_train_list,
                                            GT_list=train_list_GT,
                                            image_list=train_list,
                                            image_size=config.c1_size,
                                            batch_size=config.batch_size,
                                            load_preseg=True,
                                            num_workers=config.num_workers,
                                            mode='train',
                                            augmentation_prob=config.augmentation_prob, )
    else:
        raise ('difficult or easy')
    valid_loader = get_loader(seg_list=None,
                              GT_list=valid_list_GT,
                              image_list=valid_list,
                              image_size=config.c1_size,
                              batch_size=config.batch_size_test,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0., )

    test_loader = get_loader(seg_list=None,
                             GT_list=test_list_GT,
                             image_list=test_list,
                             image_size=config.c1_size,
                             batch_size=config.batch_size_test,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0., )
    solver = StageOneTrainer(config, train_loader, valid_loader, test_loader, logging)

    if config.mode == 'train':
        solver.train()


if __name__ == '__main__':
    config = get_args()
    main(config)
