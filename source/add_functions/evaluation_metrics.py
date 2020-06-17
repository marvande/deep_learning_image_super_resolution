import pandas as pd
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
import tensorflow as tf
from torchvision.models import vgg16_bn
import torchvision.transforms as transforms
from skimage import measure
from sklearn import metrics

from numpy import linalg as LA
import shutil

base_loss = F.l1_loss



# MSE for multi-channel:
def mse_mult_chann(imageA, imageB):
    """
    mse_mult_chann: Mean Squared Error' between the two images is the sum of the squared difference between the two images;NOTE: the two images must have the same dimension
    @input: 
    -  np.array imageA, imageB: images of type (num channel, height, width)    """
    
    num_channels = imageA.shape[0]
    m = imageA.shape[1]
    n = imageA.shape[2]
    mse_channels = []
    
    # converts to pixel range 0-255 if of range 0-1
    if np.max(imageA) <=1.0 and np.max(imageB) <=1.0:
        imageA = imageA*255
        imageB = imageB*255
        
    for i in range(num_channels):
        err = np.sum((imageA[i,:,:].astype("float") - imageB[i,:,:].astype("float")) ** 2)
        err /= float(m * n)
        mse_channels.append(err)
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return np.mean(mse_channels)

# MSE for multi-channel:
def mae_mult_chann(imageA, imageB):
    """
    mse_mult_chann: Mean Squared Error' between the two images is the sum of the squared difference between the two images;NOTE: the two images must have the same dimension
    @input: 
    -  np.array imageA, imageB: images of type (num channel, height, width)    """
    
    num_channels = imageA.shape[0]
    m = imageA.shape[1]
    n = imageA.shape[2]
    mae_channels = []
    
    # converts to pixel range 0-255 if of range 0-1
    if np.max(imageA) <=1.0 and np.max(imageB) <=1.0:
        imageA = imageA*255
        imageB = imageB*255
        
    for i in range(num_channels):
        metrics.mean_absolute_error
        mae_channels.append(metrics.mean_absolute_error(imageA[0,:,:],imageB[0,:,:] ))
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return np.mean(mae_channels)


def ssim_mult_chann(imageA, imageB):
    """
    ssim_mult_chann: evaluates the average SSIM for a multi-channel image
    @input: 
    -  np.array imageA, imageB: images of type (num channel, height, width)    """
    assert(imageA.shape == imageB.shape)
    ssim = [measure.compare_ssim(imageA[i,:,:], imageB[i,:,:]) for i in range(imageA.shape[0])]
    return np.mean(ssim)

def frob(imageA, imageB):
    """
    frob: evaluates the frobenius norm of the substraction of two arrays
    @input: 
    -  np.array imageA, imageB: images of type (num channel, height, width)    """
    assert (imageA.shape == imageB.shape)
    norm = []
    for i in range(imageA.shape[0]):
        norm.append(LA.norm(imageA[i, :, :] - imageB[i, :, :], 'fro'))
    return np.mean(norm)


def nuc(imageA, imageB):
    """
    nuc: evaluates the nuclear norm of the substraction of two arrays
    arr1, arr2: np.array of image of type (num channel, height, width)
    @input: 
    -  np.array imageA, imageB: images of type (num channel, height, width)    
    """    
    assert (imageA.shape == imageB.shape)
    norm = [LA.norm(imageA[i, :, :] - imageB[i, :, :], 'nuc') for i in range(imageA.shape[0])]
    return np.mean(norm)


def multi_chann_psnr(img1, img2):
    """
    multi_chann_psnr: evaluates the average PSNR for a multi-channel image
    @input: 
    - np.array img1, img2: images of type (num channel, height, width)
    """
    PIXEL_MAX = 255
    assert (img1.shape == img2.shape)

    # converts to pixel range 0-255 if of range 0-1
    if np.max(img1) <=1.0 and np.max(img2) <=1.0:
        img1 = img1*255
        img2 = img2*255

    psnr = []
    for i in range(len(img1[0])):
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            psnr.append(100)
        psnr.append(20 * math.log10(PIXEL_MAX / math.sqrt(mse)))
    return np.mean(psnr)

def evaluate_metrics(arr1, arr2, title):
    """
    evalute_metrics: evaluates different metrics between two arrays
    @input:
    - np.array arr1, arr2: inut arrays of shape (num channel, height, width)
    - str title
    """
    assert (arr1.shape == arr2.shape)
    #evaluate norm between MR, HR:
    FROB = frob(arr1, arr2)
    NUC = nuc(arr1, arr2)
    print(title + ': Frobenius diff: %.4f frob, Nuclear diff: %.4f' %
          (FROB, NUC))

    PSNR = measure.compare_psnr(arr1, arr2)
    MSE = mse_mult_chann(arr1, arr2 )
    NMSE = measure.compare_nrmse(arr1, arr2)
    SSIM = ssim_mult_chann(arr1, arr2)
    L1 = mae_mult_chann(arr1,arr2)
    print(title + ': MAE: %4f, MSE: %.4f, NMSE: %.4f, PSNR: %.4f, SSIM: %.4f' %
          (L1,MSE, NMSE, PSNR, SSIM))
    return MSE, NMSE, SSIM, FROB, NUC

def median_avg_df(df, res, ups = False):
    """median_avg_df: calculates the median and average of metrics
    """
    if ups == False:
        diff = ''
    else: 
        diff = res + '_'
        res = res + '_ups'
    av_metrics = pd.DataFrame(index = [res],
        data={
            'avg_MAE': np.mean(df[diff+'MAE']),
            'med_MAE': np.median(df[diff+'MAE']),
            'avg_MSE': np.mean(df[diff+'MSE']),
            'med_MSE': np.median(df[diff+'MSE']),
            'avg_NMSE': np.mean(df[diff+'NMSE']),
            'med_NMSE': np.median(df[diff+'NMSE']),
            'avg_SSIM': np.mean(df[diff+'SSIM']),
            'med_SSIM': np.median(df[diff+'SSIM']),
            'avg_PSNR': np.mean(df[diff+'PSNR']),
            'med_PSNR': np.median(df[diff+'PSNR']),
            
        })
    av_metrics = av_metrics.transpose()
    av_metrics['metrics'] = av_metrics.index
    return av_metrics
