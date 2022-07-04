# Few-Shot_Orchid_types_Classification

## Getting the code
You can download all the files in this repository by cloning this repository:  
```
git clone https://github.com/Jia-Wei-Liao/Orchid_Classification.git
```

## Download
- You can download the dataset on the Google Drive:  
https://drive.google.com/drive/folders/1iNrKzPCQRLAJy1gR0hfA7L_y-6TRB2bg?usp=sharing
- You can download the checkpoint on the Google Drive:  
https://drive.google.com/drive/folders/1fzuG2sy1vxOeA5M0BDUZJTXl3GS0wA-O?usp=sharing

## Training
To train the model, you can run this command:
```
python train.py -bs <batch size> \
                -ep <epoch> \
                --model <model> \
                --pretrain <using pretrained-weight> \
                --fold <fold of 4 fold cross validation>
                --img_size <crop and resize to img_size> \
                --loss <loss function> \
                --optim <optimizer> \
                --lr <learning rate> \
                --weight_decay <parameter weight decay> \
                --scheduler <learning rate schedule> \
                --autoaugment <use autoaugmentation> \
                --rot_degree <degree of rotation> \
                --fliplr <probabiliy of horizontal flip> \
                --noise <probabiliy of adding gaussian noise> \
                --num_workers <number worker> \
                --device <gpu id> \
                --seed <random seed>
```
- model: EfficientB4, Swin, CSwin, ConvB
- loss: CE, MCCE, FL, FLSD
- optim: SGD, Adam, AdamW, Ranger
- scheduler: step (gamma, step_size), cos

### For EfficientNet_b4 with MCCE loss
```
python train.py --model EfficientB4 --img_size 416 --loss MCCE
```

### For Swin Transformer
```
python train.py --model Swin --pretrain 1 --img_size 384 -bs 64 --lr 3e-4
```


## Testing
To test the results, you can run this command:
```
python test.py --checkpoint <XX-XX-XX-XX-XX> --topk <number of model you want to ensemble>
```

## Inference
To inference the results, you can run this command:  
- **Version 1:**
```
python inference.py --checkpoint 06-06-10-04-43 06-06-10-03-00 06-06-09-29-55 06-06-07-46-11 06-06-05-29-11 --topk 2 1 1 2 3
```
- **Version 2:**
```
python inference.py --checkpoint 05-11-23-12-52 05-12-02-24-32 05-12-05-36-53 05-12-04-58-22 05-12-01-46-14 05-11-22-34-25 05-12-03-41-16 05-12-00-29-30 06-06-11-22-42 06-06-06-00-22 --topk 5 1 2 1 2 1 1 1 1 1
```

Finally, you must convert the id to the label by runing this command:
```
python convert.py
```


## Experiment results
Before training, we use RandomResizedCrop(416), RandomHorizontalFlip(p=0.5), RandomRotation(degree=10), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), RandomNoise(p=0.1) and Autoaugmentation as preprocessing.

<table>
  <tr>
    <td>checkpoint</td>
    <td>model</td>
    <td>bs</td>
    <td>loss</td>
    <td>optimizer</td>
    <td>scheduler</td>
    <td>accuracy</td>
  </tr>
  <tr>
    <td>04-11-23-42-37</td>
    <td>EfficientNet-B4</td>
    <td>16</td>
    <td>CE</td>
    <td>AdamW (lr=1e-3)</td>
    <td>Step</td>
    <td>0.8630</td>
  </tr>
  <tr>
    <td>04-11-23-58-50</td>
    <td>EfficientNet-B4</td>
    <td>16</td>
    <td>MCCE</td>
    <td>AdamW (lr=1e-3)</td>
    <td>Step</td>
    <td>0.8584</td>
  </tr>
  <tr>
    <td>04-12-00-14-06</td>
    <td>EfficientNet-B4</td>
    <td>32</td>
    <td>FLSD</td>
    <td>AdamW (lr=1e-3)</td>
    <td>Step</td>
    <td>0.8607</td>
  </tr>
  <tr>
    <td>04-12-00-14-06</td>
    <td>EfficientNet-B4</td>
    <td>32</td>
    <td>FLSD</td>
    <td>AdamW (lr=5e-3)</td>
    <td>Step</td>
    <td>0.8402</td>
  </tr>
  <tr>
    <td>04-12-01-49-02</td>
    <td>EfficientNet-B4</td>
    <td>32</td>
    <td>CE</td>
    <td>AdamW (lr=1e-3)</td>
    <td>Step</td>
    <td>0.8402</td>
  </tr>
  <tr>
    <td>04-12-02-08-09</td>
    <td>EfficientNet-B4</td>
    <td>8</td>
    <td>CE</td>
    <td>AdamW (lr=1e-3)</td>
    <td>Step</td>
    <td>0.8447 </td>
  </tr>
  <tr>
    <td>04-12-02-09-10</td>
    <td>EfficientNet-B4</td>
    <td>16</td>
    <td>FL</td>
    <td>AdamW (lr=1e-3)</td>
    <td>Step</td>
    <td>0.8699 </td>
  </tr>
  <tr>
    <td>04-12-02-19-38</td>
    <td>EfficientNet-B4</td>
    <td>16</td>
    <td>FLSD</td>
    <td>AdamW (lr=1e-3)</td>
    <td>Step</td>
    <td>0.8470 </td>
  </tr>
  <tr>
    <td>04-12-16-01-10</td>
    <td>Swin</td>
    <td>32</td>
    <td>CE</td>
    <td>AdamW (lr=3e-4)</td>
    <td>Step</td>
    <td>0.8927 </td>
  </tr>
  <tr>
    <td>04-13-19-57-43</td>
    <td>Swin</td>
    <td>32</td>
    <td>FL</td>
    <td>AdamW (lr=3e-4)</td>
    <td>Step</td>
    <td>0.9110 </td>
  </tr>
  <tr>
    <td>04-13-19-57-43</td>
    <td>Swin</td>
    <td>64</td>
    <td>FL</td>
    <td>AdamW (lr=3e-4)</td>
    <td>Step</td>
    <td>0.9110</td>
  </tr>
  <tr>
    <td>04-19-13-58-40</td>
    <td>ConvNext-B</td>
    <td>64</td>
    <td>FL</td>
    <td>AdamW (lr=3e-4)</td>
    <td>Step</td>
    <td>0.8836</td>
  </tr>
  <tr>
    <td>04-19-21-12-43</td>
    <td>ConvNext-B</td>
    <td>64</td>
    <td>FL</td>
    <td>AdamW (lr=3e-4)</td>
    <td>Step</td>
    <td>0.8699</td>
  </tr>
  <tr>
    <td>04-19-21-12-43</td>
    <td>ConvNext-B</td>
    <td>64</td>
    <td>FLSD</td>
    <td>AdamW (lr=3e-4)</td>
    <td>Step</td>
    <td>0.8836</td>
  </tr>
  <tr>
    <td>04-27-17-28-28</td>
    <td>CSwin</td>
    <td>32</td>
    <td>FL</td>
    <td>AdamW (lr=3e-5)</td>
    <td>Step</td>
    <td>0.8676</td>
  </tr>
  <tr>
    <td>04-27-19-58-18</td>
    <td>CSwin</td>
    <td>32</td>
    <td>FL</td>
    <td>AdamW (lr=1e-4)</td>
    <td>Step</td>
    <td>0.8858</td>
  </tr>
  <tr>
    <td>04-27-21-12-57</td>
    <td>CSwin</td>
    <td>32</td>
    <td>FL</td>
    <td>AdamW (lr=8e-5)</td>
    <td>Step</td>
    <td>0.8744</td>
  </tr>
<table>

## GitHub Acknowledgement
We thank the authors of these repositories:
- MCCE: https://github.com/Kurumi233/Mutual-Channel-Loss  
- FLSD: https://github.com/torrvision/focal_calibration  
- Ranger21: https://github.com/lessw2020/Ranger21  
- AutoAugment: https://github.com/DeepVoltaire/AutoAugment  
- https://github.com/TW-yuhsi/ViT-Orchids-Classification


## Citation
```bibtex
@misc{
    title  = {orchid_classification},
    author = {Jia-Wei Liao, Yu-Hsi Chen, Jing-Lun Huang},
    url    = {https://github.com/Jia-Wei-Liao/Orchid_Classification},
    year   = {2022}
}
```
