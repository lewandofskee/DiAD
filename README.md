# [AAAI 2024] DiAD
**DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection**


[Haoyang He<sup>1#</sup>](https://scholar.google.com/citations?hl=zh-CN&user=8NfQv1sAAAAJ),
[Jiangning Zhang<sup>1,2#</sup>](https://zhangzjn.github.io),
[Hongxu Chen<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=uFT3YfMAAAAJ),
[Xuhai Chen<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=LU4etJ0AAAAJ),
[Zhishan Li<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=9g-IRLsAAAAJ),
[Xu Chen<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=1621dVIAAAAJ),
[Yabiao Wang<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=xiK4nFUAAAAJ),
[Chengjie Wang<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=fqte5H4AAAAJ),
[Lei Xie<sup>1*</sup>](https://scholar.google.com/citations?hl=zh-CN&user=7ZZ_-m0AAAAJ)

(#Equal contribution, *Corresponding author)

[<sup>1</sup>College of Control Science and Engineering, Zhejiang University](http://www.cse.zju.edu.cn/), 
[<sup>2</sup>Youtu Lab, Tencent](https://open.youtu.qq.com/#/open)

[[`Paper`](https://arxiv.org/abs/2312.06607)] 
[[`Project Page`](https://lewandofskee.github.io/projects/diad/)]

Our DiAD will also be supported in [ADer](https://github.com/zhangzjn/ADer)

## Abstract
Reconstruction-based approaches have achieved remarkable outcomes in anomaly detection. The exceptional image reconstruction capabilities of recently popular diffusion models have sparked research efforts to utilize them for enhanced reconstruction of anomalous images. Nonetheless, these methods might face challenges related to the preservation of image categories and pixel-wise structural integrity in the more practical multi-class setting. To solve the above problems, we propose a Difusion-based Anomaly Detection (DiAD) framework for multi-class anomaly detection, which consists of a pixel-space autoencoder, a latent-space Semantic-Guided (SG) network with a connection to the stable diffusion’s denoising network, and a feature-space pre-trained feature extractor. Firstly, The SG network is proposed for reconstructing anomalous regions while preserving the original image’s semantic information. Secondly, we introduce Spatial-aware Feature Fusion (SFF) block to maximize reconstruction accuracy when dealing with extensively reconstructed areas. Thirdly, the input and reconstructed images are processed by a pre-trained feature extractor to generate anomaly maps based on features extracted at different scales. Experiments on MVTec-AD and VisA datasets demonstrate the effectiveness of our approach which surpasses the state-of-the-art methods, e.g., achieving 96.8/52.6 and 97.2/99.0 (AUROC/AP) for localization and detection respectively on multi-class MVTec-AD dataset.
## 1. Installation

First create a new conda environment

    conda env create -f environment.yaml
    conda activate diad
    pip3 install timm==0.8.15dev0 mmselfsup pandas transformers openpyxl imgaug numba numpy tensorboard fvcore accimage Ninja
## 2.Dataset
### 2.1 MVTec-AD
- **Create the MVTec-AD dataset directory**. Download the MVTec-AD dataset from [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad). Unzip the file and move them to `./training/MVTec-AD/`. The MVTec-AD dataset directory should be as follows. 

```
|-- training
    |-- MVTec-AD
        |-- mvtec_anomaly_detection
            |-- bottle
                |-- ground_truth
                    |-- broken_large
                        |-- 000_mask.png
                    |-- broken_small
                        |-- 000_mask.png
                    |-- contamination
                        |-- 000_mask.png
                |-- test
                    |-- broken_large
                        |-- 000.png
                    |-- broken_small
                        |-- 000.png
                    |-- contamination
                        |-- 000.png
                    |-- good
                        |-- 000.png
                |-- train
                    |-- good
                        |-- 000.png
        |-- train.json
        |-- test.json
```

### 2.2 VisA
- **Create the VisA dataset directory**. Download the VisA dataset from [VisA_20220922.tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar). Unzip the file and move them to `./training/VisA/`. The VisA dataset directory should be as follows. 

```
|-- training
    |-- VisA
        |-- visa
            |-- candle
                |-- Data
                    |-- Images
                        |-- Anomaly
                            |-- 000.JPG
                        |-- Normal
                            |-- 0000.JPG
                    |-- Masks
                        |--Anomaly 
                            |-- 000.png        
        |-- visa.csv
```

## 3. Finetune the Autoencoders
- Finetune the Autoencoders first by downloading the pretrained Autoencoders from [kl-f8.zip](https://ommer-lab.com/files/latent-diffusion/kl-f8.zip). Move it to `./models/autoencoders.ckpt`.
And finetune the model with running


`python finetune_autoencoder.py`

- Once finished the finetuned model is under the folder `./lightning_logs/version_x/checkpoints/epoch=xxx-step=xxx.ckpt`.
Then move it to the folder with changed name `./models/mvtec_ae.ckpt`. The same finetune process on VisA dataset.
- If you use the given pretrained autoencoder model, you can go step 4 to build the model.

| Autoencoder        | Pretrained Model                                                                                 |
|--------------------|--------------------------------------------------------------------------------------------------|
| MVTec First Stage Autoencoder | [mvtecad_fs](https://drive.google.com/file/d/1vDfywjGqoWRHMxj-5fifujK29_XyHuCQ/view?usp=sharing) |
| VisA First Stage Autoencoder  | [visa_fs](https://drive.google.com/file/d/1zycpAbWwIVodwTo0Bh1oK8xKliuTT3ul/view?usp=sharing)    |

## 4. Build the model
- We use the pre-trianed stable diffusion v1.5, the finetuned autoencoders and the Semantic-Guided Network to build the full needed model for training.
The stable diffusion v1.5 could be downloaded from ["v1-5-pruned.ckpt"](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main). Move it under the folder `./models/v1-5-pruned.ckpt`. 
Then run the code to get the output model `./models/diad.ckpt`.

`python build_model.py`


## 5. Train
- Training the model by simply run

`python train.py`
- Batch size, learning rate, data path, gpus, and resume path could be easily edited in `train.py`.


## 6. Test
The output of the saved checkpoint could be saved under `./val_ckpt/epoch=xxx-step=xxx.ckpt`For evaluation and visualization, set the checkpoint path `--resume_path` and run the following code:

`python test.py --resume_path ./val_ckpt/epoch=xxx-step=xxx.ckpt`

The images are saved under `./log_image/, where
- `xxx-input.jpg` is the input image.
- `xxx-reconstruction.jpg` is the reconstructed image through autoencoder without diffusion model.
- `xxx-features.jpg` is the feature map of the anomaly score.
- `xxx-samples.jpg` is the reconstructed image through the autoencoder and diffusion model.
- `xxx-heatmap.png` is the heatmap of the anomaly score.

## Citation
If you find this code useful, don't forget to star the repo and cite the paper:
```
@misc{he2023diad,
      title={DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection},
      author={Haoyang He and Jiangning Zhang and Hongxu Chen and Xuhai Chen and Zhishan Li and Xu Chen and Yabiao Wang and Chengjie Wang and Lei Xie},
      year={2023},
      eprint={2312.06607},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Acknowledgements
We thank the great works [UniAD](https://github.com/zhiyuanyou/UniAD), [LDM](https://github.com/CompVis/latent-diffusion) and [ControlNet](https://github.com/lllyasviel/ControlNet) for providing assistance for our research.
