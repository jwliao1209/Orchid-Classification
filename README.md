# Orchid_Classification

## Getting the code
You can download all the files in this repository by cloning this repository:  
```
git clone https://github.com/Jia-Wei-Liao/Orchid_Classification.git
```

## Training
To train the model, you can run this command:
```
python train.py
```
- model: EfficientB4, MCEfficientB4
- loss: CE, MCCE, FL, FLSD
- optim: SGD, Adam, AdamW, Ranger
- scheduler: step, cos


## Test
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
    <td>RandomResizedCrop(416),<br>RandomHorizontalFlip(p=0.5),<br>RandomRotation(degree=10),<br>Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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
    <td>RandomResizedCrop(416),<br>RandomHorizontalFlip(p=0.5),<br>RandomRotation(degree=10),<br>Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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
    <td>RandomResizedCrop(416),<br>RandomHorizontalFlip(p=0.5),<br>RandomRotation(degree=10),<br>Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    <td>84.93 (ep=107)</td>
    <td> </td>
  </tr>
</table>

## Reference
[1] FLSD: https://github.com/torrvision/focal_calibration  
[2] Ranger21: https://github.com/lessw2020/Ranger21  
[3] AutoAugment: https://github.com/DeepVoltaire/AutoAugment  
