import pandas as pd
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
import tensorflow as tf
from torchvision.models import vgg16_bn
import torchvision.transforms as transforms
from skimage import measure
from numpy import linalg as LA
import shutil

from  add_functions.evaluation_metrics import *
from  add_functions.model_create_data_functions import *

class PyTMinMaxScaler(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __call__(self, tensor):
        for ch in tensor:
            scale = 1.0 / (ch.max() - ch.min())
            ch.mul_(scale).sub_(ch.min())
        return tensor

scaler = PyTMinMaxScaler()

def plot_double(arr1, arr2, title1, title2):
    figure, ax = plt.subplots(1, 2, figsize=(20, 20))
    ax[0].imshow(arr1, cmap='gray')
    ax[0].set_title(title1)
    ax[1].imshow(arr2, cmap='gray')
    ax[1].set_title(title2)


def plot_triple(arr1, arr2, arr3, title1, title2, title3):
    figure, ax = plt.subplots(1, 3, figsize=(20, 20))
    ax[0].imshow(arr1, cmap='gray')
    ax[0].set_title(title1)
    ax[1].imshow(arr2, cmap='gray')
    ax[1].set_title(title2)
    ax[2].imshow(arr3, cmap='gray')
    ax[2].set_title(title3)
    
def plot_several_tests(learn, LR_list, MR_list, HR_list):
    for i in range(len(LR_list[0:10])):

        file_name, patient, location = file_inf(str(LR_list[i]))
        HR_tens = scaler(open_image(HR_list[i]).data)
        LR_tens = scaler(open_image(LR_list[i]).data)
        MR_tens = scaler(open_image(MR_list[i]).data)


        #resize LR to same size as HR:
        img_lr = PIL.Image.open(LR_list[i]).resize(
            (672, 502), resample=PIL.Image.BICUBIC).convert('RGB')

        #resize MR to same size as HR:
        img_lr = PIL.Image.open(MR_list[i]).resize(
            (672, 502), resample=PIL.Image.BICUBIC).convert('RGB')
        MR_tens = scaler(trans1(img_lr))

        # Prediction of model for ups LR:
        p_lr, img_pred_lr, b_lr = learn.predict(Image(LR_tens))
        img_pred_lr = scaler(img_pred_lr)

        # Prediction of model for ups MR:
        p_mr, img_pred_mr, b_mr = learn.predict(Image(MR_tens))
        img_pred_mr = scaler(img_pred_mr)

        print('Patient: {}, Location: {}'.format(patient, location))
        plot_triple(img_pred_mr.numpy()[1, :, :],
                    HR_tens.numpy()[1, :, :],
                    img_pred_lr.numpy()[1, :, :], 'Pred MR', 'gt HR', 'pred LR')

def plot_several_tests_2(learn, LR_list, MR_list, path_hr):
    for i in range(10):
        
        file_str = str(LR_list[i])
        # get file string
        pattern = "LR\/(.*?)\.tif"
        substring = re.search(pattern, file_str).group(1)
        file_name = substring+'.tif'
        
        HR_tens = scaler(open_image(path_hr/file_name).data)
        LR_tens = scaler(open_image(LR_list[i]).data)
        MR_tens = scaler(open_image(MR_list[i]).data)
        plot_triple(MR_tens.numpy()[1, :, :],
                    HR_tens.numpy()[1, :, :],
                    LR_tens.numpy()[1, :, :], 'Pred MR', 'gt HR', 'pred LR')
        
def plot_metrics(df,res, sim):
    # report average and median MSE, NMSE, SSIM: 
    fig = plt.figure(figsize = (15, 6))
    ax = fig.add_subplot(1, 5, 1)
    ax.plot(df['MAE'],  '-x', label='pred HR vs GT')
    ax.plot(df[res+'_MAE'],'-x', label = 'int HR vs GT')
    ax.axhline(np.median(df[res+'_MAE']),color = 'r')
    ax.axhline(np.median(df['MAE']),color = 'b')
    ax.set_title('MAE')
    plt.xlabel('files')
    plt.ylabel('MAE')
    ax.legend()

    ax = fig.add_subplot(1, 5, 2)
    ax.plot(df['MSE'],  '-x', label='pred HR vs GT')
    ax.plot(df[res+'_MSE'],'-x', label = 'int HR vs GT')
    ax.axhline(np.median(df[res+'_MSE']),color = 'r')
    ax.axhline(np.median(df['MSE']),color = 'b')
    ax.set_title('MSE')
    plt.xlabel('files')
    plt.ylabel('MSE')
    ax.legend()

    ax = fig.add_subplot(1, 5, 3)
    ax.plot(df['NMSE'],  '-x')
    ax.plot(df[res+'_NMSE'],'-x')
    ax.axhline(np.median(df[res+'_NMSE']),color = 'r')
    ax.axhline(np.median(df['NMSE']),color = 'b')
    ax.set_title('NMSE')
    plt.xlabel('files')
    plt.ylabel('NMSE')

    ax = fig.add_subplot(1, 5, 4)
    ax.plot(df['SSIM'],  '-x')
    ax.plot(df[res+'_SSIM'],'-x')
    ax.axhline(np.median(df[res+'_SSIM']),color = 'r')
    ax.axhline(np.median(df['SSIM']),color = 'b')
    ax.set_title('SSIM')
    plt.xlabel('files')
    plt.ylabel('SSIM')

    ax = fig.add_subplot(1, 5, 5)
    ax.plot(df['PSNR'],  '-x')
    ax.plot(df[res+'_PSNR'],'-x')
    ax.axhline(np.median(df[res+'_PSNR']),color = 'r')
    ax.axhline(np.median(df['PSNR']),color = 'b')
    ax.set_title('PSNR')
    plt.xlabel('files')
    plt.ylabel('PSNR')

    plt.suptitle(res+' vs GT')
    plt.savefig('images/intHRvsGT_sim{}_'.format(sim)+res+'.png')
    
def plot_metrics_feat(df1,df2, res, sim):
    # report average and feature and gram losses: 
    fig = plt.figure(figsize = (18, 10))
    ax = fig.add_subplot(2, 4, 1)
    ax.plot(df1['overall_loss'],  '-x', label='pred HR vs GT')
    ax.plot(df2['overall_loss'],'-x', label = 'int HR vs GT')
    ax.axhline(np.median(df1['overall_loss']),color = 'b')
    ax.axhline(np.median(df2['overall_loss']),color = 'r')
    ax.set_title('overall_loss')
    plt.ylabel('overall_loss')
    ax.legend()

    ax = fig.add_subplot(2, 4, 2)
    ax.plot(df1['pixel_loss'],  '-x', label='pred HR vs GT')
    ax.plot(df2['pixel_loss'],'-x', label = 'int HR vs GT')
    ax.axhline(np.median(df1['pixel_loss']),color = 'b')
    ax.axhline(np.median(df2['pixel_loss']),color = 'r')
    ax.set_title('pixel_loss')
    plt.ylabel('pixel_loss')
    ax.legend()

    ax = fig.add_subplot(2, 4, 3)
    ax.plot(df1['feat0_loss'],  '-x', label='pred HR vs GT')
    ax.plot(df2['feat0_loss'],'-x', label = 'int HR vs GT')
    ax.axhline(np.median(df1['feat0_loss']),color = 'b')
    ax.axhline(np.median(df2['feat0_loss']),color = 'r')
    ax.set_title('feat0_loss')
    plt.ylabel('feat0_loss')
    ax.legend()

    ax = fig.add_subplot(2, 4, 4)
    ax.plot(df1['feat1_loss'],  '-x', label='pred HR vs GT')
    ax.plot(df2['feat1_loss'],'-x', label = 'int HR vs GT')
    ax.axhline(np.median(df1['feat1_loss']),color = 'b')
    ax.axhline(np.median(df2['feat1_loss']),color = 'r')
    ax.set_title('feat1_loss')
    plt.ylabel('feat1_loss')
    ax.legend()

    ax = fig.add_subplot(2, 4, 5)
    ax.plot(df1['feat2_loss'],  '-x', label='pred HR vs GT')
    ax.plot(df2['feat2_loss'],'-x', label = 'int HR vs GT')
    ax.axhline(np.median(df1['feat2_loss']),color = 'b')
    ax.axhline(np.median(df2['feat2_loss']),color = 'r')
    ax.set_title('feat2_loss')
    plt.ylabel('feat2_loss')
    plt.xlabel('files')
    ax.legend()
    
    ax = fig.add_subplot(2, 4, 6)
    ax.plot(df1['gram0_loss'],  '-x', label='pred HR vs GT')
    ax.plot(df2['gram0_loss'],'-x', label = 'int HR vs GT')
    ax.axhline(np.median(df1['gram0_loss']),color = 'b')
    ax.axhline(np.median(df2['gram0_loss']),color = 'r')
    ax.set_title('gram0_loss')
    plt.ylabel('gram0_loss')
    plt.xlabel('files')
    ax.legend()

    
    ax = fig.add_subplot(2, 4, 7)
    ax.plot(df1['gram1_loss'],  '-x', label='pred HR vs GT')
    ax.plot(df2['gram1_loss'],'-x', label = 'int HR vs GT')
    ax.axhline(np.median(df1['gram1_loss']),color = 'b')
    ax.axhline(np.median(df2['gram1_loss']),color = 'r')
    ax.set_title('gram1_loss')
    plt.ylabel('gram1_loss')
    plt.xlabel('files')

    ax.legend()
    
    ax = fig.add_subplot(2, 4, 8)
    ax.plot(df1['gram2_loss'],  '-x', label='pred HR vs GT')
    ax.plot(df2['gram2_loss'],'-x', label = 'int HR vs GT')
    ax.axhline(np.median(df1['gram2_loss']),color = 'b')
    ax.axhline(np.median(df2['gram2_loss']),color = 'r')
    ax.set_title('gram2_loss')
    plt.ylabel('gram2_loss')
    ax.legend()
    plt.xlabel('files')


    plt.suptitle(res+' vs GT')
    plt.savefig('images/intHRvsGT_feat_loss_sim{}_'.format(sim)+res+'.png')
