import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from src.config import *
from src.dataset import get_test_loader
from src.trainer import compute_acc
from src.utils import *
from sklearn.metrics import classification_report
import ttach as tta


def get_topk_ckpt(weight_path, topk):
    topk_ckpt = []

    for c, t in zip(weight_path, topk):
        weight_list = sorted(
            glob.glob(os.path.join('checkpoint', c, 'weight', '*.pth')),
            key=lambda x: float(x[-10:-4]), reverse=True)

        topk_ckpt += weight_list[:t]

    return topk_ckpt


def init_model(models, ckpts, device):
    for m, c in zip(models, ckpts):
        m.load(c)
        m.to(device)
        m.eval()

    return models


def test(args):
    test_loader, _ = get_test_loader(args)
    models = [get_model(args) for i in range(sum(args.topk))]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    topk_ckpt = get_topk_ckpt(args.checkpoint, args.topk)
    models = init_model(models, topk_ckpt, device)

    total_num, correct = 0, 0
    test_bar = tqdm(test_loader, desc=f'Testing')
    pred_list = []
    label_list = []

    with torch.no_grad():
        for batch_data in test_bar:
            image, label = batch_data
            image = image.to(device)
            label = label.to(device)

            tta_transforms = tta.Compose([
                tta.HorizontalFlip(),
                tta.FiveCrops(args.img_size, args.img_size)
                ])

            pred = []
            for m in models:
                tta_model = tta.ClassificationTTAWrapper(m, tta_transforms)
                pred.append(torch.nn.functional.softmax(tta_model(image), dim=1))

            pred = torch.mean(torch.stack(pred, dim=0), dim=0)
            acc = compute_acc(pred, label)
            num = image.shape[0]
            total_num += num
            correct += acc * num
            pred = pred.argmax(dim=1)
            pred_list += list(pred.cpu().numpy())
            label_list += list(label.cpu().numpy())

            del image, label, pred
            mean_acc = correct / total_num
            test_bar.set_postfix({
                'acc': f"{mean_acc:.5f}"
            })
        test_bar.close()

    print(classification_report(label_list, pred_list, target_names=[str(i) for i in range(219)]))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default=['05-12-05-36-53'],
                        nargs='+', help='weight path')
    parser.add_argument('--topk', type=int,
                        default=[3],
                        nargs='+', help='weight of topk accuracy')
    
    args = parser.parse_args()
    config = load_json(os.path.join('checkpoint', args.checkpoint[0], 'config.json'))
    args = argparse.Namespace(**vars(args), **config)
    test(args)
