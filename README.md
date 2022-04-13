# Orchid_Classification

## Getting the code
You can download all the files in this repository by cloning this repository:  
```
git clone https://github.com/Jia-Wei-Liao/Orchid_Classification.git
```

## Training
To train the model, you can run this command:
```
python train.py -bs <batch size> \
                -ep <epoch> \
                --model <model> \
                --pretrain <using pretrained-weight> \
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
- model: EfficientB4, Swin
- loss: CE, MCCE, FL, FLSD
- optim: SGD, Adam, AdamW, Ranger
- scheduler: step (gamma, step_size), cos

### For EfficientNet_b4 with MCCE loss
```
python --model EfficientB4 --img_size 416 --loss MCCE
```

### For Swin Transformer
```
python --model Swin --img_size 384 --lr 3e-4
```


## Testing
To test the results, you can run this command:
```
python test.py --checkpoint <XX-XX-XX-XX-XX> --weight <ep=XXXX-acc=0.XXXX.pth>
```


## Experiment results
<table>
  <tr>
    <td>Checkpoint</td>
    <td>Model</td>
    <td>Batch size</td>
    <td>Epochs</td>
    <td>Loss</td>
    <td>Optimizer</td>
    <td>Scheduler</td>
    <td>Augmentation</td>
    <td>Best val acc</td>
    <td>test acc</td>
  </tr>
  <tr>
    <td>04-11-18-49-39</td>
    <td>EfficientNet-b4</td>
    <td>32</td>
    <td>200</td>
    <td>CE</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>RandomResizedCrop(416),<br>RandomHorizontalFlip(p=0.5),<br>RandomRotation(degree=10),<br>Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))</td>
    <td>85.62 (ep=75) </td>
    <td>84.27 </td>
  </tr>
  <tr>
    <td>04-11-20-01-09</td>
    <td>EfficientNet-b4</td>
    <td>32</td>
    <td>200</td>
    <td>FLSD</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>RandomResizedCrop(416),<br>RandomHorizontalFlip(p=0.5),<br>RandomRotation(degree=10),<br>Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))</td>
    <td>86.07 (ep=160)</td>
    <td>85.85</td>
  </tr>
    <tr>
    <td>04-11-21-51-39</td>
    <td>EfficientNet-b4</td>
    <td>32</td>
    <td>200</td>
    <td>MCCE</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>RandomResizedCrop(416),<br>RandomHorizontalFlip(p=0.5),<br>RandomRotation(degree=10),<br>Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))</td>
    <td>84.93 (ep=107)</td>
    <td>83.79 </td>
  </tr>
  <tr>
    <td>04-11-23-42-37</td>
    <td>EfficientNet-b4</td>
    <td>16</td>
    <td>200</td>
    <td>CE</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>RandomResizedCrop(416),<br>RandomHorizontalFlip(p=0.5),<br>RandomRotation(degree=10),<br>Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))</td>
    <td>86.30 (ep=36)</td>
    <td> </td>
  </tr>
  <tr>
    <td>04-11-23-58-50</td>
    <td>EfficientNet-b4</td>
    <td>16</td>
    <td>200</td>
    <td>MCCE</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>RandomResizedCrop(416),<br>RandomHorizontalFlip(p=0.5),<br>RandomRotation(degree=10),<br>Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))<br>RandomNoise(p=0.1)</td>
    <td>85.84 (ep=92)</td>
    <td> </td>
  </tr>
  <tr>
    <td>04-12-02-09-10</td>
    <td>EfficientNet-b4</td>
    <td>16</td>
    <td>200</td>
    <td>FL</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>RandomResizedCrop(416),<br>RandomHorizontalFlip(p=0.5),<br>RandomRotation(degree=10),<br>Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))<br>RandomNoise(p=0.1)</td>
    <td>86.99 (ep=92)</td>
    <td> </td>
  </tr>
</table>


## GitHub Acknowledgement
We thank the authors of these repositories:
- MCCE: https://github.com/Kurumi233/Mutual-Channel-Loss  
- FLSD: https://github.com/torrvision/focal_calibration  
- Ranger21: https://github.com/lessw2020/Ranger21  
- AutoAugment: https://github.com/DeepVoltaire/AutoAugment  


## Citation
```
@misc{
    title  = {orchid_classification},
    author = {Jia-Wei Liao, Yu-Hsi Chen, Jing-Lun Huang, Kuok-Tong Ng},
    url    = {https://github.com/Jia-Wei-Liao/Orchid_Classification},
    year   = {2022}
}
```
