# Orchid_Classification

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
  </tr>
  <tr>
    <td>04-11-00-15-58</td>
    <td>EfficientNet-b4</td>
    <td>16</td>
    <td>100</td>
    <td>CE</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>RandomResizedCrop(416),<br>RandomHorizontalFlip(p=0.5),<br>RandomRotation(p=10),<br>Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    <td>82.19 (ep=75)</td>
  </tr>
</table>
