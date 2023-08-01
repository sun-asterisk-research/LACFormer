# LACFormer: Toward accurate and efficient polyp segmentation
This repository contains the official Pytorch implementation of training & evaluation code for LACFormer.

### Environment
- Creating a virtual environment in terminal: `conda create -n LACFormer python=3.8.16`
- Install `CUDA 11.3` and `pytorch 1.8.1`: `conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge`
- Install other requirements: `pip install -r requirements.txt`
### Dataset
Downloading necessary data:
For `Experiment` in our paper: 
- Download testing dataset, move it into `Dataset/` and extract the file zip, which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view).
- Download training dataset, move it into `Dataset/` and extract the file zip, which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view).

    
### Training
Download MiT's pretrained [weights](https://drive.google.com/drive/folders/1-7Su1ehH6QzmBHQ42DvYv997L_lSv_Gv?usp=sharing) on ImageNet-1K, and put them in a folder `pretrained/`.
Config hyper-parameters in `mcode/config.py` and run `train.py` for training:
```
python train.py
```
Here is an example in [Google Colab](https://colab.research.google.com/drive/1VkcuPj_NBw7vpCtZT5gKAUFrZ20IdBkZ?usp=sharing)
### Evaluation
After training, evaluation will be done automatically
### Checkpoint
The checkpoint for LACFormer-L can be downloaded from [here](https://drive.google.com/file/d/1XcYz5Spvq_SnB4cQJj0BXRg0RNDrByq9/view?usp=sharing)
