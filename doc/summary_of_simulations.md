# Image super resolution, summary of simulations:

## Original model:
Input images:
| Model  |  LR |  MR |  HR | # train images | # val images | Normalization | 
|---|---|---|---|---|---|---|
| Orginal  (PETS)|   (3,96,128) |   (3, 256, 342) |   ? |  ? | ? |Imagenet stats
| Simulation 1  (MRI)| (3,96,128) |   (3, 256, 342) | (3, 1004, 1344) |   427 | 47| Imagenet stats |
| Simulation 2 (MRI)| (3, 500, 669)|  (3, 1004, 1344)  | (3, 1004, 1344)  |  1388 | 154|Imagenet stats |
| Simulation 3 (MRI) (cut in 4)| (3, 250, 334)| (3, 502, 672)  |  (3, 502, 672)  |  5552 | 616 |Imagenet stats |
| Simulation 4 (MRI) (cut in 4)| (3, 250, 334)| (3, 502, 672)  |  (3, 502, 672)  |  5552 | 616 |Imagenet stats |
| Simulation 5 (MRI) (cut in 4)| (3, 250, 334)| (3, 502, 672)  |  (3, 502, 672)  |  5552 | 616 |Image stats|
| Simulation 6 (MRI) (cut4, tiff)| (3, 250, 334)| (3, 502, 672)  |  (3, 502, 672)  |  1851 | 205 |Image stats|
| Simulation 7 (MRI) (cut4, tiff, w/o transf)| (3, 250, 334)| (3, 502, 672)  |  (3, 502, 672)  |  1851 | 205 |Image stats|
| Simulation 8 (MRI) (cut4, tiff, w/o transf)| (3, 125, 167)| (3, 502, 672)  |  (3, 502, 672)  |  1851 | 205 |Image stats|
| Simulation 9 (MRI) (cut4, tiff, with transf)| (3, 125, 167)| (3, 250, 334)  |  (3, 502, 672)  |  1851 | 205 |Imagenet stats|




### Training: 
- architecture = unet_learner
- data = transformed with a bunch of transformations (see notebook) and normalized to imagenet_stats 

#### Phase 1:
**Note**: Validation and training loss values correspond to the end of the 10th epoch. 
**Phase 1a**:
| Model  | x |  y  | Batch size | Num epochs | Weight decay | learning rate |Train loss |  Val loss|
|---|---|---|---|---|---|---|---|---|
| Orginal  | (3, 128, 128) |  (3, 128, 128) | 32 | 10 | 1e-3 | slice(0.01)| 3.356 |	3.187| 
| Simulation 1  | (3, 128, 128) |  (3, 128, 128) | 32 | 10 | 1e-3 | slice(0.01)| 1.541 |1.168| 
| Simulation 2 |  (3, 500, 500) |   (3, 500, 500)|4 | 10 | 1e-3 | slice(0.01)| 0.5513 |	0.5490 |
| Simulation 3 (cut in 4)|  (3, 250, 250)|  (3, 250, 250)|20 | 10 | 1e-3 | slice(0.01)| 0.6505 |0.6240 
| Simulation 4 (cut in 4)|  (3, 250, 334)|  (3, 250, 334)|15 | 10 | 1e-3 | slice(0.01)| 0.605637 |0.587562 
| Simulation 5 (cut in 4)|  (3, 250, 334)|  (3, 250, 334)|15 | 10 | 1e-3 | slice(0.01)| 0.6043 | 0.5753
| Simulation 6 (cut4, tiff)|  (3, 250, 334)|  (3, 250, 334)|15 | 10 | 1e-3 | slice(0.01)| 0.449800 |0.501512
| Simulation  7 (MRI) (cut4, tiff, w/o transf)|  (3, 250, 334)|  (3, 250, 334)|15 | 10 | 1e-3 | slice(0.01)| 0.351183 |0.346088
| Simulation  8 (MRI) (cut4, tiff, w/o transf)|  (3, 250, 334)|  (3, 250, 334)|15 | 10 | 1e-3 | slice(0.01)| 0.619969 |0.604122
| Simulation  9 (MRI) (cut4, tiff, with transf)|  (3, 250, 334)|  (3, 250, 334)|15 | 10 | 1e-3 | slice(0.01)| 0.591649 |0.525496





**Phase 1b**: 
| Model  |   x |  y  | Batch size | Num epochs | Weight decay | learning rate |Train loss |  Val loss |
|---|---|---|---|---|---|---|---|---|
| Orginal  |   (3, 128, 128) |  (3, 128, 128) | 32 | 10 | 1e-3 | slice(1e-5, 1e-3)|3.261|	3.124 |
| Simulation 1  |   (3, 128, 128) |  (3, 128, 128) | 32 | 10 | 1e-3 | slice(1e-5, 1e-3)|1.1189|	1.090 |
| Simulation 2 | (3, 500, 500) |   (3, 500, 500)|4 | 10 | 1e-3 | slice(1e-5, 1e-3) |0.5476 |	0.5245
| Simulation 3 (cut in 4)|  (3, 250, 250)|  (3, 250, 250)|20 | 10 | 1e-3 | slice(1e-5, 1e-3)| 0.6066 |0.5747 |
| Simulation 4 (cut in 4)|  (3, 250, 334)|  (3, 250, 334)|15 | 10 | 1e-3 | slice(1e-5, 1e-3)| 0.594584 |0.571241 |
| Simulation 5 (cut in 4)|  (3, 250, 334)|  (3, 250, 334)|15 | 10 | 1e-3 | slice(1e-5, 1e-3)| 0.5854 |0.5634 |
| Simulation 6 (cut4, tiff)|  (3, 250, 334)|  (3, 250, 334)|15 | 10 | 1e-3 | slice(1e-5, 1e-3)| 0.425064 |0.509859 |
| Simulation  7 (MRI) (cut4, tiff, w/o transf)|  (3, 250, 334)|  (3, 250, 334)|15 | 10 | 1e-3 |slice(1e-5, 1e-3)| 0.341797 |0.341707 
| Simulation  8 (MRI) (cut4, tiff, w/o transf)|  (3, 250, 334)|  (3, 250, 334)|15 | 10 | 1e-3 |slice(1e-5, 1e-3)| 0.580875 |0.576627
| Simulation  9 (MRI) (cut4, tiff, with transf)|  (3, 250, 334)|  (3, 250, 334)|15 | 10 | 1e-3 |slice(1e-5, 1e-3)| 0.541038 |0.494553 




#### Phase 2:
**Note**: Validation and training loss values correspond to the end of the 10th epoch. 

**Phase 2a**: 
| Model  |   x |  y  | Batch size | Num epochs | Weight decay | learning rate |Train loss |  Val loss |
|---|---|---|---|---|---|---|---|---|
| Orginal  |    (3, 256, 256)|  (3, 256, 256)| 12 | 10 | 1e-3 | slice(1e-3)|2.061 |	2.080
| Simulation 1  |    (3, 256, 256)|  (3, 256, 256)| 12 | 10 | 1e-3 | slice(1e-3)|1.105 |	1.056
| Simulation 2 |  (3, 1004, 1004) |  (3, 1004, 1004) | 1 | 10 | 1e-3 | slice(1e-3)|0.6276|	0.651
| Simulation 3 (cut in 4)|  (3, 502, 502)|  (3, 502, 502)| 5 | 10 | 1e-3 | slice(1e-3)| 0.7236 |0.7507 |
| Simulation 4 (cut in 4)|  (3, 502, 672)|  (3, 502, 672)| 4 | 10 | 1e-3 | slice(1e-3)|0.710625|0.746289 |
| Simulation 5 (cut in 4)|  (3, 502, 672)|  (3, 502, 672)| 4 | 10 | 1e-3 | slice(1e-3)|0.7023|0.7366 |
| Simulation 6 (cut4, tiff)|  (3, 502, 672)|  (3, 502, 672)| 4 | 10 | 1e-3 | slice(1e-3)|0.639402|0.847978 |
| Simulation 7 (MRI) (cut4, tiff, w/o transf)|  (3, 502, 672)|  (3, 502, 672)| 4 | 10 | 1e-3 | slice(1e-3)|0.689234|0.680575 |
| Simulation 8 (MRI) (cut4, tiff, w/o transf)|  (3, 502, 672)|  (3, 502, 672)| 4 | 10 | 1e-3 | slice(1e-3)|0.839572|0.830458 |
| Simulation 9 (MRI) (cut4, tiff, with transf)|  (3, 502, 672)|  (3, 502, 672)| 4 | 10 | 1e-3 | slice(1e-3)|0.713906|0.776364 |





**Phase 2b**: 
| Model  |   x |  y  | Batch size | Num epochs | Weight decay | learning rate | Train loss |  Val loss |
|---|---|---|---|---|---|---|---|---|
| Orginal  | (3, 256, 256) |  (3, 256, 256) | 12 | 10 | 1e-3 | slice(1e-6, 1e-4)| 2.0526 |	2.0644
| Simulation 1  | (3, 256, 256) |  (3, 256, 256) | 12 | 10 | 1e-3 | slice(1e-6, 1e-4)| 1.062 |	1.034
| Simulation 2 | (3, 1004, 1004) |  (3, 1004, 1004) |  | 10 | 1e-3 | slice(1e-6, 1e-4)| 0.628 |	0.6488
| Simulation 3 (cut in 4)|  (3, 502, 502)|  (3, 502, 502)| 5 | 10 | 1e-3 | slice(1e-6, 1e-4)| ? |	? |
| Simulation 4 (cut in 4)|  (3, 502, 672)|  (3, 502, 672)| 4 | 10 | 1e-3 | slice(1e-6, 1e-4)| 0.694823 |	0.721953 |
| Simulation 5 (cut in 4)|  (3, 502, 672)|  (3, 502, 672)| 4 | 10 | 1e-3 | slice(1e-6, 1e-4)| 0.68534 |	0.71327|
| Simulation 6 (cut4, tiff)|  (3, 502, 672)|  (3, 502, 672)| 4 | 10 | 1e-3 | slice(1e-6, 1e-4)| 0.614185 |	0.792894|
| Simulation 7 (MRI) (cut4, tiff, w/o transf)|  (3, 502, 672)|  (3, 502, 672)| 4 | 10 | 1e-3 | slice(1e-6, 1e-4)| 0.676916 |	0.665724|
| Simulation 8 (MRI) (cut4, tiff, w/o transf)|  (3, 502, 672)|  (3, 502, 672)| 4 | 10 | 1e-3 | slice(1e-6, 1e-4)| 0.830131 |	0.817874|
| Simulation 9 (MRI) (cut4, tiff, with transf)|  (3, 502, 672)|  (3, 502, 672)| 4 | 10 | 1e-3 | slice(1e-6, 1e-4)| 0.557891 |	0.520142|




Model | Average MSE | Average NMSE | Average SSIM | Average MSE LR | Average NMSE LR | Average SSIM LR| 
|---|---|---|---|---|---|---|
| Simulation 4 (MRI) (cut in 4)|0.000873|0.0298464|0.9134315| 0.000288 | 0.0181| 0.9720| 
| Simulation 6 (MRI) (cut4, tiff)|0.001371|0.035364|0.884909|0.000111| 0.011021|0.992113|
| Simulation 7 (MRI) (cut4, tiff, w/o transf)|0.001425|0.036337|0.877673|0.000167| 0.013528|0.985824|

Model | Median MSE | Median NMSE | Median SSIM | Median MSE LR | Median NMSE LR | Median SSIM LR| 
|---|---|---|---|---|---|---|
| Simulation 4 (MRI) (cut in 4)|0.000565|0.025679 | 0.939164| 0.000258 | 0.016857| 0.975474| 
| Simulation 6 (MRI) (cut4, tiff)|0.000582|0.026722|0.940734|.000081| 0.009626|0.994210| 
| Simulation 7 (MRI) (cut4, tiff, w/o transf)|0.000622|0.027623|0.934779|0.000119| 0.011631|0.988207|


**Feature loss**: 
Create feature loss: 
- pre_trained model from pytorch vgg16 : VGG 16-layer model (configuration “D”) with batch normalization “Very Deep Convolutional Networks For Large-Scale Image Recognition”
- Architecture: ([5, 12, 22, 32, 42],
 [ReLU(inplace=True),
	ReLU(inplace=True),
	ReLU(inplace=True),
	ReLU(inplace=True),
	ReLU(inplace=True)])
- base loss = l1 loss
- feature loss = ? 


### Issues: 
- Creating training data: ideally 6 channel tiff images but had to convert to three channel images --> Several Deep Learning Framework Libraries (.e.g PyTorch's TorchVision), Image Processing libraries (.e.g skimage) and machine learning data augmentation libraries (e.g. Augmenter) internally rely on the PIL to perform image transformations. You can have a situation where in a Semantic Segmentation problem, you have a 3-ch RGB input image, but have a 10-ch mask with each individual channels containing a binary mask of the classes found in the RGB image.If you use PIL to perform any type of transform, e.g. rescale, it truncates a multi-channel image to a 3-channel image.
- Training data very similar according to MSE, NMSE, SSIM (need to evaluate differently) 
- in depth analysis of pixels
- report: why not a GAN ? unstable and does not converge quickly
