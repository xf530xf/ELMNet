# ELMNet
ELMNet: Extremely Lightweight Mamba Network for Biological Image Segmentation



**0. Main Environments.** </br>
The environment installation procedure can be followed by [VM-UNet](https://github.com/JCruan519/VM-UNet), or by following the steps below (python=3.8):</br>
```
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

**1. Datasets.** </br>
Data preprocessing environment installation (python=3.7):
```
conda create -n tool python=3.7
conda activate tool
pip install h5py
conda install scipy==1.2.1  # scipy1.2.1 only supports python 3.7 and below.
pip install pillow
```

*A. ISIC2017* </br>
1. Download the ISIC 2017 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `/data/dataset_isic17/`. </br>
2. Run `Prepare_ISIC2017.py` for data preparation and dividing data to train, validation and test sets. </br>

*B. ISIC2018* </br>
1. Download the ISIC 2018 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `/data/dataset_isic18/`. </br>
2. Run `Prepare_ISIC2018.py` for data preparation and dividing data to train, validation and test sets. </br>


*C. Prepare your own dataset* </br>
1. The file format reference is as follows. (The image is a 24-bit png image. The mask is an 8-bit png image. (0 pixel dots for background, 255 pixel dots for target))
- './your_dataset/'
  - images
    - 0000.png
    - 0001.png
  - masks
    - 0000.png
    - 0001.png
  - Prepare_your_dataset.py
2. In the 'Prepare_your_dataset.py' file, change the number of training sets, validation sets and test sets you want.</br>
3. Run 'Prepare_your_dataset.py'. </br>

**2. Train the ELMNet.** </br>
You can simply run the following command to start training, or download the weights file based on this [issue](https://github.com/wurenkai/UltraLight-VM-UNet/issues/38) before training.
```
python train.py
```
- After trianing, you could obtain the outputs in './results/' </br>

**3. Test the ELMNet.**  
First, in the test.py file, you should change the address of the checkpoint in 'resume_model'.
```
python test.py
```
- After testing, you could obtain the outputs in './results/' </br>
