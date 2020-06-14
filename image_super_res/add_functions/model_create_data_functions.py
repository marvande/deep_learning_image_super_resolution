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
import random

from  add_functions.evaluation_metrics import *
from  add_functions.plot_fx_model import *
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# transformations to use in model if want to avoid overfitting. 
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

def resize_one(fn, i, path, size, rel_to_path):
    """resize_one: resizes images to input size and saves them in path, 
    quality is lowered as to get LR images. 
    @input:
    - fn: filename
    - i: iteration for parallel resising
    - PosixPath path: path to save resized images
    - (int, int, int) size: size to resize to 
    - PosixPath rel_to_path: path where to get the images
    """
    #destination to save new images:
    dest = path / fn.relative_to(rel_to_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    #open image:
    img = PIL.Image.open(fn)
    #resize image: 
    img = img.resize(size, resample=PIL.Image.BICUBIC).convert('RGB')
    #save new image: 
    img.save(dest, quality=60)
    
def get_data(bs, size, test_path, path_lr, path_hr):
    """
    get_data: creates training, validation and test data from LR and HR.
    HR and LR are resized to size. 
    @input: 
    - int bs: batch size
    - (int, int, int) size: size for data
    - PosixPath test_path: path with test data
    - PosixPath path_lr: path to LR images
    - PosixPath path_hr: path to HR images
    """
    #splits data into validation and training
    src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)
    #adds label (here HR images)
    data = src.label_from_func(lambda x: path_hr / x.relative_to(path_lr))

    #resizes images, add test set and normalize
    """Note: as we feed it to a pretrained resnet model, expect input images normalized in the same wayi.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]"""    
    data = data.transform(tfms,size=size, tfm_y=True).add_test(ImageList.from_folder(test_path), tfms=None, tfm_y=False).databunch(bs=bs).normalize(imagenet_stats,do_y=True)
    #number of channels
    data.c = 3
    return data


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

#transforms to tensor:
trans1 = transforms.ToTensor()
#transforms to pillow image:
trans = transforms.ToPILImage()

def file_inf(file_str):
    # get file string
    pattern = "test\/(.*?)\.tif"
    substring = re.search(pattern, file_str).group(1)
    file_name = substring+'.tif'
    
    # get patient number:
    pattern = "test\/(.*?)_"
    patient = re.search(pattern, file_str).group(1)
    
    # get location number:
    pattern2 = "\[(.*?)\]"
    location = '['+re.search(pattern2,file_name).group(1)+']'
    
    return file_name, patient, location

def create_pred_int(learn, path_lr, path_hr, save_int, save_pred):
    # saves predictions and interpolated versions for all images
    LR_list = ImageList.from_folder(path_lr).items
    HR_list = ImageList.from_folder(path_hr).items
    
    for i in range(len(HR_list)):
        file_name, patient, location = file_inf(str(LR_list[i]))

        #resize LR to size of HR
        img_lr = PIL.Image.open(LR_list[i]).resize(
            (672, 502), resample=PIL.Image.BICUBIC).convert('RGB')
        LR_tens = scaler(trans1(img_lr))

        #HR img:
        HR_tens = open_image(HR_list[i]).data
        
        # Prediction of model for ups LR:
        p_lr, img_pred_lr, b_lr = learn.predict(Image(LR_tens))
        
        # Assert reconstructed HR has same shape as ground truth HR:
        assert (list(img_pred_lr.shape) == list(HR_tens.shape))  
        
        # save created images
        img_lr.save(save_int/file_name)
        trans(scaler(img_pred_lr)).save(save_pred/file_name)
            
def eval_test_images_df(learn, path_lr, path_hr, res, indices):
    MAE, MSE, NMSE, SSIM, FROB, PSNR = [], [],[], [], [], []
    LR_MAE,LR_MSE, LR_NMSE, LR_SSIM, LR_FROB, LR_PSNR = [],[], [], [], [], [], 
    file_names, locations, patients, phenotypes_list = [], [], [], []

    LR_list = ImageList.from_folder(path_lr).items
    HR_list = ImageList.from_folder(path_hr).items
    
    for i in indices:
        file_name, patient, location = file_inf(str(LR_list[i]))
        file_names.append(file_name)
        patients.append(patient)
        locations.append(location)
        
        #resize LR to size of HR
        img_lr = PIL.Image.open(LR_list[i]).resize(
            (672, 502), resample=PIL.Image.BICUBIC).convert('RGB')
        LR_tens = scaler(trans1(img_lr))

        #HR img:
        HR_tens = open_image(HR_list[i]).data
        
        # Prediction of model for ups LR:
        p_lr, img_pred_lr, b_lr = learn.predict(Image(LR_tens))
        
        plot_triple(LR_tens.numpy()[1, :, :],
                    HR_tens.numpy()[1, :, :],img_pred_lr.numpy()[1, :, :], 
                        'Int. '+res, 'gt HR', 'Pred '+res)
        # Assert reconstructed HR has same shape as ground truth HR:
        assert (list(img_pred_lr.shape) == list(HR_tens.shape))
        MAE.append(mse_mult_chann(HR_tens.numpy(), scaler(img_pred_lr).numpy()))
        MSE.append(mse_mult_chann(HR_tens.numpy(), scaler(img_pred_lr).numpy()))
        NMSE.append(measure.compare_nrmse(HR_tens.numpy(),
                                         scaler(img_pred_lr).numpy()))
        SSIM.append(ssim_mult_chann(HR_tens.numpy(), scaler(img_pred_lr).numpy()))
        FROB.append(frob(HR_tens.numpy(), scaler(img_pred_lr).numpy()))
        PSNR.append(multi_chann_psnr(HR_tens.numpy(), scaler(img_pred_lr).numpy()))
        
        LR_MAE.append(mse_mult_chann(LR_tens.numpy(), scaler(img_pred_lr).numpy()))
        LR_MSE.append(mse_mult_chann(LR_tens.numpy(), scaler(img_pred_lr).numpy()))
        LR_NMSE.append(
            measure.compare_nrmse(LR_tens.numpy(), scaler(img_pred_lr).numpy()))
        LR_SSIM.append(ssim_mult_chann(LR_tens.numpy(), scaler(img_pred_lr).numpy()))
        LR_FROB.append(frob(LR_tens.numpy(), scaler(img_pred_lr).numpy()))
        LR_PSNR.append(multi_chann_psnr(LR_tens.numpy(), scaler(img_pred_lr).numpy()))

    return pd.DataFrame(
        data={
            'file': file_names,
            'patient': patients,
            'location': locations,
            'MAE':MAE,
            'MSE': MSE,
            'NMSE': NMSE,
            'SSIM': SSIM,
            'FROB': FROB,
            'PSNR':PSNR,
            res + '_MAE': LR_MAE,
            res + '_MSE': LR_MSE,
            res + '_NMSE': LR_NMSE,
            res + '_SSIM': LR_SSIM,
            res + '_FROB': LR_FROB,
            res + '_PSNR': LR_PSNR,
        })

def gram_matrix(x):
    """
    Gram matrix of a set of vectors in an inner 
    product space is the Hermitian matrix of inner products
    """
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)

def losses_df(data_set, vgg_m, blocks):

    feat_loss_testing = FeatureLoss_testing(vgg_m, blocks[2:5], [5,15,2])

    pixel_loss, gram0_loss, gram1_loss, gram2_loss = [], [], [], [],
    sum_loss, feat0_loss, feat1_loss, feat2_loss = [], [], [], []
    for i in range(20):

        #Feature loss between intMR and HR:
        x_mr = data_set.one_batch(ds_type=DatasetType.Valid)[0][i].numpy()
        arr_mr = np.ndarray((1, 3, 502, 672))
        arr_mr[0] = x_mr
        x_mr = torch.from_numpy(arr_mr).float().to(device)

        y_mr = data_set.one_batch(ds_type=DatasetType.Valid)[1][i].numpy()
        arr_mr = np.ndarray((1, 3, 502, 672))
        arr_mr[0] = y_mr
        y_mr = torch.from_numpy(arr_mr).float().to(device)

        over_loss, losses = feat_loss_testing.forward(x_mr, y_mr)
        sum_loss.append(over_loss.item())
        pixel_loss.append(losses['pixel'].item())
        feat0_loss.append(losses['feat_0'].item())
        feat1_loss.append(losses['feat_1'].item())
        feat2_loss.append(losses['feat_2'].item())
        gram0_loss.append(losses['gram_0'].item())
        gram1_loss.append(losses['gram_1'].item())
        gram2_loss.append(losses['gram_2'].item())

    return pd.DataFrame(
            data={
                'overall_loss':sum_loss,
                'pixel_loss': pixel_loss,
                'feat0_loss': feat0_loss,
                'feat1_loss': feat1_loss,
                'feat2_loss': feat2_loss,
                'gram0_loss': gram0_loss,
                'gram1_loss': gram1_loss,
                'gram2_loss': gram2_loss
            })

class FeatureLoss_testing(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        #feat losses
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        #gram: 
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses),self.metrics 
    
    def __del__(self): self.hooks.remove()
