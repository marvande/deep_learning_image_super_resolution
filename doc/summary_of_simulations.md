# Summary of simulations:

## Original model:
Input images:
| Model  |  LR |  MR |  HR | # train images | # val images |
|---|---|---|---|---|---|
| Orginal  (PETS)|   (3,96,128) |   (3, 256, 342) |   ? |  ? |
| Simulation 1  (MRI)| (3,96,128) |   (3, 256, 342) | (3, 1004, 1344) |   427 | 47| 
| Simulation 2 (MRI)| (3, 500, 669)|  (3, 1004, 1344)  | (3, 1004, 1344)  |  1388 | 154|
| Simulation 3 (MRI) (cut in 4)| (3, 250, 334)| (3, 502, 672)  |  (3, 502, 672)  |  1388 | 154 |

#### Training: 
- architecture = unet_learner
- data = transformed with a bunch of transformations (see notebook) and normalized to imagenet_stats 

##### Phase 1:
**Note**: Validation and training loss values correspond to the end of the 10th epoch. 
**Phase 1a**:
| Model  | x |  y  | Batch size | Num epochs | Weight decay | learning rate |Train loss |  Val loss|
|---|---|---|---|---|---|---|---|---|
| Orginal  | (3, 128, 128) |  (3, 128, 128) | 32 | 10 | 1e-3 | slice(0.01)| 3.356 |	3.187| 
| Simulation 1  | (3, 128, 128) |  (3, 128, 128) | 32 | 10 | 1e-3 | slice(0.01)| 1.541 |1.168| 
| Simulation 2 |  (3, 500, 500) |   (3, 500, 500)|4 | 10 | 1e-3 | slice(0.01)| 0.5513 |	0.5490 |
| Simulation 3 (cut in 4)|  (3, 250, 250)|  (3, 250, 250)|20 | 10 | 1e-3 | slice(0.01)| 0.6505 |	0.6240 |

**Phase 1b**: 
| Model  |   x |  y  | Batch size | Num epochs | Weight decay | learning rate |Train loss |  Val loss |
|---|---|---|---|---|---|---|---|---|
| Orginal  |   (3, 128, 128) |  (3, 128, 128) | 32 | 10 | 1e-3 | slice(1e-5, 1e-3)|3.261|	3.124 |
| Simulation 1  |   (3, 128, 128) |  (3, 128, 128) | 32 | 10 | 1e-3 | slice(1e-5, 1e-3)|1.1189|	1.090 |
| Simulation 2 | (3, 500, 500) |   (3, 500, 500)|4 | 10 | 1e-3 | slice(1e-5, 1e-3) |0.5476 |	0.5245
| Simulation 3 (cut in 4)|  (3, 250, 250)|  (3, 250, 250)|20 | 10 | 1e-3 | slice(1e-5, 1e-3)| 0.6066 |0.5747 |

##### Phase 2:
**Note**: Validation and training loss values correspond to the end of the 10th epoch. 
**Phase 2a**: 
| Model  |   x |  y  | Batch size | Num epochs | Weight decay | learning rate |Train loss |  Val loss |
|---|---|---|---|---|---|---|---|---|
| Orginal  |    (3, 256, 256)|  (3, 256, 256)| 12 | 10 | 1e-3 | slice(1e-3)|2.061 |	2.080
| Simulation 1  |    (3, 256, 256)|  (3, 256, 256)| 12 | 10 | 1e-3 | slice(1e-3)|1.105 |	1.056
| Simulation 2 |  (3, 1004, 1004) |  (3, 1004, 1004) | 1 | 10 | 1e-3 | slice(1e-3)|0.6276|	0.651
| Simulation 3 (cut in 4)|  (3, 502, 502)|  (3, 502, 502)| 5 | 10 | 1e-3 | slice(1e-3)| 0.7236 |0.7507 |
**Phase 2b**: 
| Model  |   x |  y  | Batch size | Num epochs | Weight decay | learning rate | Train loss |  Val loss |
|---|---|---|---|---|---|---|---|---|
| Orginal  | (3, 256, 256) |  (3, 256, 256) | 12 | 10 | 1e-3 | slice(1e-6, 1e-4)| 2.0526 |	2.0644
| Simulation 1  | (3, 256, 256) |  (3, 256, 256) | 12 | 10 | 1e-3 | slice(1e-6, 1e-4)| 1.062 |	1.034
| Simulation 2 | (3, 1004, 1004) |  (3, 1004, 1004) |  | 10 | 1e-3 | slice(1e-6, 1e-4)| 0.628 |	0.6488
| Simulation 3 (cut in 4)|  (3, 502, 502)|  (3, 502, 502)| 5 | 10 | 1e-3 | slice(1e-6, 1e-4)| ? |	? |

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




