from skimage import measure
import torchvision.transforms as transforms
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
import tensorflow as tf
from torchvision.models import vgg16_bn


tfms = ([
    RandTransform(tfm=TfmCrop(crop_pad),
                  kwargs={
                      'row_pct': (0, 1),
                      'col_pct': (0, 1),
                      'padding_mode': 'reflection'
                  },
                  p=1.0,
                  resolved={},
                  do_run=True,
                  is_random=True,
                  use_on_y=True),
    RandTransform(tfm=TfmPixel(flip_lr),
                  kwargs={},
                  p=0.5,
                  resolved={},
                  do_run=True,
                  is_random=True,
                  use_on_y=True),
    RandTransform(tfm=TfmCoord(symmetric_warp),
                  kwargs={'magnitude': (-0.2, 0.2)},
                  p=0.75,
                  resolved={},
                  do_run=True,
                  is_random=True,
                  use_on_y=True),
    RandTransform(tfm=TfmLighting(brightness),
                  kwargs={'change': (0.4, 0.6)},
                  p=0.75,
                  resolved={},
                  do_run=True,
                  is_random=True,
                  use_on_y=True),
    RandTransform(tfm=TfmLighting(contrast),
                  kwargs={'scale': (0.8, 1.25)},
                  p=0.75,
                  resolved={},
                  do_run=True,
                  is_random=True,
                  use_on_y=True)
], [
    RandTransform(tfm=TfmCrop(crop_pad),
                  kwargs={},
                  p=1.0,
                  resolved={},
                  do_run=True,
                  is_random=True,
                  use_on_y=True)
])

# MSE for multi-channel:
def mse_mult_chann(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    num_channels = imageA.shape[0]
    m = imageA.shape[1]
    n = imageA.shape[2]
    mse_channels = []
    for i in range(num_channels):
        err = np.sum((imageA[i,:,:].astype("float") - imageB[i,:,:].astype("float")) ** 2)
        err /= float(m * n)
        mse_channels.append(err)
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return np.mean(mse_channels)


def ssim_mult_chann(imageA, imageB):
    assert(imageA.shape == imageB.shape)
    ssim = [measure.compare_ssim(imageA[i,:,:], imageB[i,:,:]) for i in range(imageA.shape[0])]
    return np.mean(ssim)

# Takes as input images as ndarray
def compare_images_metrics(img_test,img_true,img_LR, title):
    # compute the mean squared error and structural similarity
    # index for the images
    #mse = measure.compare_mse(img_true, img_test)
    mse = mse_mult_chann(img_true, img_test)
    nmse = measure.compare_nrmse(img_true, img_test)
    ssim = ssim_mult_chann(img_true, img_test)
    
    print("MSE: %.8f, NMSE: %.8f, SSIM : %.4f" % (mse, nmse, ssim))

    # setup the figure
    fig = plt.figure(figsize = (40,40))

    # show the second image
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(img_test[0,:,:])
    ax.set_title('Predicted HR', fontsize=30)
    plt.axis('off')
    
    # show first image
    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(img_true[0,:,:])
    ax.set_title('Original HR', fontsize=30)
    plt.axis('off')

    # show first image
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(img_true[0,:,:])
    ax.set_title('upsampled LR', fontsize=30)
    plt.axis('off')
   
    # show the images
    print(title)
    plt.show()

# input image of type 'Image'
def plot_single_image(img, title, plot_size):
    fig = plt.figure(figsize=plot_size)
    ax = fig.add_subplot(1, 1, 1)
    plt.title(title)
    plt.imshow(img[0,:,:])
    
def plot_3_images(lr_resized, im_HR_gt, img_pred, plt_size):
    # setup the figure
    fig = plt.figure(figsize=plt_size)

    # show LR:
    ax = fig.add_subplot(3, 1, 1)
    plt.imshow(lr_resized[0,:,:])
    ax.set_title('LR of shape {} (resized to HR shape)'.format(list(lr_resized.shape)))

    # show ground truth HR: 
    ax = fig.add_subplot(3, 1, 2)
    plt.imshow(im_HR_gt[0,:,:])
    ax.set_title('HR ground thruth of shape {}'.format(list(im_HR_gt.shape)))

    # show predicted HR image:
    ax = fig.add_subplot(3, 1, 3)
    plt.imshow(img_pred[0,:,:])
    ax.set_title('Reconstructed HR of shape {}'.format(list(img_pred.shape)))