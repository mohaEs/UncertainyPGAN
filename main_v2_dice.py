
import torch.utils.data as data
import os.path
import numpy as np
import random
import torch
import torchvision.transforms as transforms
import os

#from torchsummary import summary
import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageOps
from sklearn import metrics
import tqdm

from torchvision.utils import save_image

from src.networks import CasUNet_3head 
from src.networks import UNet_3head
from src.networks import NLayerDiscriminator
from src.utils_Moha import train_i2i_UNet3headGAN
from src.utils_Moha import train_i2i_Cas_UNet3headGAN
import src.losses


class CustomDataset_train(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_path, image_target_path, transform = None):
        self.df = pd.read_csv(csv_path)
        self.image_path = image_path
        self.image_target_path = image_target_path
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        filename = str(self.df.loc[index, 'filename'])
        image1 = PIL.Image.open(os.path.join(self.image_path, filename))
        image_target = PIL.Image.open(os.path.join(self.image_target_path, filename[:-3]+'png'))
        
        s = image_target.getextrema()
        # print('==>', s[1])
        image_target = image_target.point(lambda i: i * (255/s[1]))

        if self.transform is not None:
            image1 = self.transform(image1)
            image_target = self.transform(image_target)

        return image1, image_target, filename#, label, int(ID)



class CustomDataset_test(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_path, transform = None):
        self.df = pd.read_csv(csv_path)
        self.image_path = image_path
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        filename = str(self.df.loc[index, 'filename'])
        image1 = PIL.Image.open(os.path.join(self.image_path, filename))

        if self.transform is not None:
            image1 = self.transform(image1)

        return image1, filename#, label, int(ID)





if __name__ == '__main__':



    img_size = 256
    batch_size = 2

    num_epochs_primeGANs = 5
    num_epochs_subGAN_1st = 2
    num_epochs_subGAN_2nd = 2
    
    dice_weight_primeGANs = 5
    dice_weight_subGAN_1st = 3
    dice_weight_subGAN_2nd = 1



    if torch.cuda.is_available():
        device = torch.device('cuda')
        dtype = torch.cuda.FloatTensor
    else:
        device = torch.device('cpu')
        dtype = torch.float

    print('==> Device: ', device)

    # device = 'cuda'
    # dtype = torch.cuda.FloatTensor

    path_checkpoint='./ckpt'

    if os.path.isdir(path_checkpoint) == False:
        os.makedirs(path_checkpoint)

    path_train_csv = './data/train.csv'
    path_train_image = './data/images'
    path_train_image_target = './data/soft_disc'

    path_test_csv = './data/test.csv'
    path_test_image = './data/images'
    path_test_image_target = './data/soft_disc'

    path_save_image_result =  './results'
    if os.path.isdir(path_save_image_result) == False:
        os.makedirs(path_save_image_result)




    transform_train = transforms.Compose([
                transforms.Resize((int(img_size), int(img_size))),                
                transforms.Grayscale(num_output_channels=1),  
                transforms.ToTensor(),                 
    ])

    
    transform_test = transforms.Compose([
                transforms.Resize((int(img_size), int(img_size))),   
                transforms.Grayscale(num_output_channels=1),  
                transforms.ToTensor()
    ])


    trainset= CustomDataset_train(path_train_csv, path_train_image, path_train_image_target, transform_train)
    testset = CustomDataset_test(path_test_csv, path_test_image, transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, num_workers=4, shuffle = True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = 1, num_workers=1, shuffle = False)


    ###############
    #''' train primary GAN'''
    ###############

    print('==> primary GAN training ...' )

    netG_A = CasUNet_3head(1,1)
    netD_A = NLayerDiscriminator(1, n_layers=4)
    netG_A, netD_A = train_i2i_UNet3headGAN(
        netG_A, netD_A,
        train_loader, test_loader,
        dtype = dtype,
        device= device,
        num_epochs=num_epochs_primeGANs,
        init_lr = 1e-5,
        ckpt_path=os.path.join(path_checkpoint,'i2i_0_UNet3headGAN'),
        dice_weight = dice_weight_primeGANs
        )

    ###############
    #''' train subsequent GAN #1'''
    ###############

    print('==> sebsequent GAN #1 training ...' )

    # first load the prior Generators 
    netG_A0 = CasUNet_3head(1,1)
    netG_A0.load_state_dict(torch.load(os.path.join(path_checkpoint,'i2i_0_UNet3headGAN_eph' + str(num_epochs_primeGANs-1) + '_G_A.pth')))

    #initialize the current GAN
    netG_A1 = UNet_3head(4,1)
    netD_A = NLayerDiscriminator(1, n_layers=4)

    #train the cascaded framework
    list_netG_A, list_netD_A = train_i2i_Cas_UNet3headGAN(
        [netG_A0, netG_A1], [netD_A],
        train_loader, test_loader,
        dtype = dtype,
        device= device,
        num_epochs=num_epochs_subGAN_1st,
        init_lr=1e-5,
        ckpt_path=os.path.join(path_checkpoint,'i2i_1_UNet3headGAN'),
        dice_weight = dice_weight_subGAN_1st
    )    

    ###############
    #''' train subsequent GAN #2'''
    ###############

    print('==> sebsequent GAN #2 training ...' )

    # first load the prior Generators 
    netG_A0 = CasUNet_3head(1,1)
    netG_A0.load_state_dict(torch.load(os.path.join(path_checkpoint,'i2i_0_UNet3headGAN_eph' + str(num_epochs_primeGANs-1) + '_G_A.pth')))
    netG_A1 = UNet_3head(4,1)
    netG_A1.load_state_dict(torch.load(os.path.join(path_checkpoint,'i2i_1_UNet3headGAN_eph' + str(num_epochs_subGAN_1st-1) + '_G_A.pth')))

    #initialize the current GAN
    netG_A2 = UNet_3head(4,1)
    netD_A = NLayerDiscriminator(1, n_layers=4)

    #train the cascaded framework
    list_netG_A, list_netD_A = train_i2i_Cas_UNet3headGAN(
        [netG_A0, netG_A1, netG_A2], [netD_A],
        train_loader, test_loader,
        dtype=torch.cuda.FloatTensor,
        device='cuda',
        num_epochs = num_epochs_subGAN_2nd,
        init_lr=1e-5,
        ckpt_path=os.path.join(path_checkpoint,'i2i_2_UNet3headGAN'),
        dice_weight = dice_weight_subGAN_2nd
    )

    ###############
    #''' test'''
    ###############

    print('==> Testing ...' )

    netG_A0 = CasUNet_3head(1,1)
    netG_A0.load_state_dict(torch.load(os.path.join(path_checkpoint,'i2i_0_UNet3headGAN_eph' + str(num_epochs_primeGANs-1) + '_G_A.pth')))
    netG_A1 = UNet_3head(4,1)
    netG_A1.load_state_dict(torch.load(os.path.join(path_checkpoint,'i2i_1_UNet3headGAN_eph' + str(num_epochs_subGAN_1st-1) + '_G_A.pth')))
    netG_A2 = UNet_3head(4,1)
    netG_A2.load_state_dict(torch.load(os.path.join(path_checkpoint,'i2i_2_UNet3headGAN_eph' + str(num_epochs_subGAN_2nd-1) + '_G_A.pth')))

    for i, batch in enumerate(test_loader):  

        xA = batch[0]#.to(device).type(dtype)
        filename = batch[1]        
        filename = filename[0]
        #print(type(filename[0]))
        #calc all the required outputs
        rec_B, rec_alpha_B, rec_beta_B = netG_A0(xA)

        xch = torch.cat([rec_B, rec_alpha_B, rec_beta_B, xA], dim=1)
        rec_B, rec_alpha_B, rec_beta_B = netG_A1(xch)

        xch = torch.cat([rec_B, rec_alpha_B, rec_beta_B, xA], dim=1)
        rec_B, rec_alpha_B, rec_beta_B = netG_A2(xch)

        #print(type(rec_B))
        #print(rec_B)
        save_image(rec_B, os.path.join(path_save_image_result, filename[:-3]+'png'))
        # rec_B.detach()


