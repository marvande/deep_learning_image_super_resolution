from skimage import measure
import torchvision.transforms as transforms
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
import tensorflow as tf
from torchvision.models import vgg16_bn



# Takes as input images as ndarray
def compare_images_metrics(img_true, img_test):
    # compute the mean squared error and structural similarity
    # index for the images
    mse = measure.compare_mse(img_true, img_test)
    nmse = measure.compare_nrmse(img_true, img_test)
    s1 = measure.compare_ssim(img_true[0,:,:], img_test[0,:,:])
    s2 = measure.compare_ssim(img_true[1,:,:], img_test[1,:,:])
    s3 = measure.compare_ssim(img_true[2,:,:], img_test[2,:,:])
    print('Predicted HR (left) VS original HR (right)')
    print("MSE: %.8f, NMSE: %.8f, SSIMS for each channel: %.4f, %.4f, %.4f" % (mse, nmse, s1,s2,s3))

    # setup the figure
    fig = plt.figure(figsize = (40,40))
    # show first image
    ax = fig.add_subplot(3, 2, 1)
    plt.imshow(img_true[0,:,:], cmap = plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(3, 2, 2)
    plt.imshow(img_test[0,:,:], cmap = plt.cm.gray)
    plt.axis("off")
    # show first image
    ax = fig.add_subplot(3, 2, 3)
    plt.imshow(img_true[1,:,:], cmap = plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(3, 2, 4)
    plt.imshow(img_test[1,:,:], cmap = plt.cm.gray)
    plt.axis("off")
    
    # show first image
    ax = fig.add_subplot(3, 2, 5)
    plt.imshow(img_true[1,:,:], cmap = plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(3, 2, 6)
    plt.imshow(img_test[1,:,:], cmap = plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.show()

# input image of type 'Image'
def plot_single_image(img, title, plot_size):
    fig = plt.figure(figsize=plot_size)
    ax = fig.add_subplot(1, 1, 1)
    plt.title(title)
    show_image(img, ax=ax)    
    
def plot_3_images(lr_resized, im_HR_gt, img_pred, plt_size):
    # setup the figure
    fig = plt.figure(figsize=plt_size)

    # show LR:
    ax = fig.add_subplot(3, 1, 1)
    show_image(lr_resized, ax=ax)
    ax.set_title('LR of shape {} (resized to HR shape)'.format(list(lr_resized.shape)))

    # show ground truth HR: 
    ax = fig.add_subplot(3, 1, 2)
    im_HR_gt.show(ax=ax)
    ax.set_title('HR ground thruth of shape {}'.format(list(im_HR_gt.shape)))

    # show predicted HR image:
    ax = fig.add_subplot(3, 1, 3)
    show_image(img_pred, interpolation='nearest', ax=ax)
    ax.set_title('Reconstructed HR of shape {}'.format(list(img_pred.shape)))