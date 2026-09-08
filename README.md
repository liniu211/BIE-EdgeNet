# BIE-EdgeNet
Difference-Verified Bilateral Interaction and Edge Restoration for Building Change Detection in High-Resolution Remote Sensing Imagery

# Requirement
- Python 3.8
- Pytorch 1.10.1
- timm 0.4.12
- einops
- opencv-python (cv2)
- tqdm

# Dataset
The dataset should consist of bi-temporal image pairs with pixel-level change labels and edge labels. Each sample provides the pre-event image, the post-event image, the binary change mask (label), and the corresponding edge mask (label_edge).

For building change detection (BCD), download the [LEVIR-CD](https://justchenhao.github.io/LEVIR/), [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html). Crop each image into 256x256 patches and prepare the dataset into the following structure.

An example sample of the CDCD dataset (3 image pairs with labels and edge labels) is provided in the `CDCD/` folder of this repository for format reference.

Organize the dataset into the following structure:

```
/E:/Data_AI
  /CDCD
    /A                  pre-event images
    /B                  post-event images
    /label              binary change masks (0/255)
    /label_edge         edge masks generated from labels
    /list
      train.txt
      val.txt
      test.txt
  /LEVIR-CD
    ...
  /WHU-CD
    ...
```

The edge masks in `label_edge/` can be generated from the binary change masks via:
```
python edge_making.py
```
Then register your dataset path in `data_config.py`, e.g. `root_dir = r'E:\Data_AI\LEVIR-CD'`.

The `rqx-*` series datasets (plain bi-temporal pairs with binary labels, no edge labels required) are also registered in `data_config.py`: `rqx-WHU-CD`, `rqx-LEVIR-CD`, `rqx-CLCD-CD`, `rqx-GVLM-CD`, `rqx-LBFD-CD`, and `rqx-S2Looking` (via `CDDatasetAbs`).

# Supported Models

All 31 models below are registered in `models/networks.py` (`define_G`) and are selected via the `--net_G` argument.

**Proposed models (ours)**

| `net_G` | Description | Loss |
|---|---|---|
| `BIE_EdgeNet` | BIE-EdgeNet (bilateral interaction + edge restoration) | `ce` |
| `BIE_EdgeNet_DINO` | BIE-EdgeNet with DINO backbone | `ce` |
| `BIE_ChangeDINO` | BIE variant with ChangeDINO (MobileNetV2) backbone | `DINO_Loss` |

**OpenCD baselines (auto-load ImageNet backbone pretraining from `models/pretrained/`)**

| `net_G` | Description | Loss |
|---|---|---|
| `OpenCD_SNUNet` | SNUNet-CD (ECAM), trained from scratch | `ce` |
| `OpenCD_BIT` | BIT r18 (ResNetV1c-18 + BIT head) | `ce` |
| `OpenCD_ChangeFormer` | ChangeFormer MiT-b0 | `ce` |
| `OpenCD_CGNet` | Change Guiding Network (VGG16-BN backbone) | `CGNet_BCE` |
| `OpenCD_Changer` | Changer r18 (stage-2 spatial exchange + stage-3/4 channel exchange) | `ce` |
| `Change3D` | Change3D X3D-L, official BCD protocol (sigmoid probability output) | `BCEDiceLoss` |

**Published contrast methods**

| `net_G` | Description | Loss |
|---|---|---|
| `BIT` | BIT with transformer encoder (dd8) and ResNet stages | `ce` |
| `ChangeFormer` | ChangeFormer V6 (transformer encoder + conv decoder) | `ce` |
| `ChangeMamba` | ChangeMamba-BCD (VMamba backbone) | `MambaBCDLoss` |
| `ChangeDINO` | ChangeDINO (MobileNetV2 + FPN) | `DINO_Loss` |
| `HSANet` | HSANet | `HSANet_Loss` |
| `B2CNet` | B2CNet | `B2CNetLoss` |
| `EGPNet` | Edge-Guidance Network | EGP focal + dice |
| `EATDer` | EATDer | `BCE` (pos_weight=4, dual head) |
| `EGRCNN` | Edge-Guided RCNN (U-Net mtask) | `EGRCNN_Loss` |
| `BGSNet` | BGSNet (PVTv2-b1) | `AutomaticWeightedLoss` |
| `AERNet` | AERNet | `AERNet_Loss` |
| `ICIFNet` | ICIFNet | `ce` |
| `DMINet` | DMINet | `ce` |
| `ChangeViT` | ChangeViT (ViT-S) | `ce` |
| `VcT` | VcT (ResNet34-based reliable transformer) | `ce` |
| `EGCTNet` | EGCTNet | `ce` |
| `RCDT` | RCDT | `RCDT_MultiScale_Loss` |
| `CTDFormer` | CTDFormer | `ce` |

**BIT-family base variants**

| `net_G` | Description | Loss |
|---|---|---|
| `BIT_ResNet` | BIT with ResNet-18 backbone | `ce` |
| `BIT_ResNet1` | BIT with alternative ResNet backbone | `ce` |
| `base_transformer_pos_s4` | BASE transformer (stage 4, learned pos) | `ce` |
| `base_transformer_pos_s4_dd8_dedim8` | BASE transformer (dd8, decoder dim 8) | `ce` |

For the `rqx-*` datasets, multiple models can be trained sequentially with a unified recipe (bs16 / AdamW lr 1e-4 / linear decay) via `train_redo_jobs.py`; each job writes to `checkpoints/<net_G>-<data_name>/`. If a `last_ckpt.pt` already exists there, training automatically resumes from it and runs up to `--max_epochs`:

```powershell
python train_redo_jobs.py --jobs "OpenCD_BIT@rqx-WHU-CD,OpenCD_Changer@rqx-WHU-CD" --max_epochs 100
```

# Working Example

0. Environment
```bash
cd BIE-EdgeNet
conda create -n BIE python=3.8
conda activate BIE
# PyTorch 1.10.1 + CUDA 10.2
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.2 -c pytorch
pip install timm==0.4.12 einops opencv-python tqdm
```

1. Training the Model.

(1) Prepare the Dataset. Make sure your dataset is structured as described above and adjust the `--data_name` and `--project_name` arguments in `main_cd.py` to point to your dataset. (2) Training Script. To train the BIE-EdgeNet model, run the following command:
```
python main_cd.py
```
Note: The model parameters are set to batch_size of 16, epochs of 200, and learning_rate of 1e-4 (AdamW), which you can modify according to your needs. (3) Model Output. At the end of training, the model's parameters will be saved in the `./checkpoints/<project_name>/` folder, including `best_ckpt.pt` and `last_ckpt.pt`. You can use the saved model for inference.

2. Testing the Model.

(1) Prepare the Dataset. Make sure your dataset is structured as described above and adjust the `--data_name` argument in the testing script to point to your testing data folder. (2) Testing Script. To test the trained model on the test dataset, run the following command:
```
python eval_cd.py
```
Note: The `eval_cd.py` script loads the pre-trained model from `./checkpoints/<project_name>/best_ckpt.pt`. You can select the appropriate pre-trained model parameters as input. (3) Model Output. The model's prediction maps and evaluation metrics (F1, IoU, OA) are saved in the `./vis/<project_name>/` folder.

Additional Note: Ablation studies can be performed by switching the `mode` argument in `main_cd.py` (`ALL`, `CNN_Tr`, `CNN_Tr_BIE`, `CNN_Tr_Edge`, `Edge`).

# Acknowledgments

The  dataset is constructed based on the LEVIR-CD and WHU-CD change detection datasets. Thanks for their excellent works!
```
@article{chen2020spatial,
  title={A spatial-temporal attention-based method and a new dataset for remote sensing image change detection},
  author={Chen, Hao and Shi, Zhenwei},
  journal={Remote Sensing},
  volume={12},
  number={10},
  pages={1662},
  year={2020},
  publisher={MDPI}
}
```

```
@article{ji2018fully,
  title={Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set},
  author={Ji, Shunping and Wei, Shiqing and Lu, Meng},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={57},
  number={1},
  pages={574--586},
  year={2018},
  publisher={IEEE}
}
```

# Citation
If you use this code for your research, please cite our paper.

```
@article{BIE-EdgeNet,
  title={BIE-EdgeNet: Difference-Prior Cross Attention and Edge Fusion for Building Change Detection in High-Resolution Imagery},
  author={Qiao Zhang and Yuhuan Zhang and Guangliang Cheng and Zhoufeng Wang and Qianxi Rong and Yang Shen and Xuxiong Xu},
  journal={...},
  year={202X},
}
```
