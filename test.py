import os
import argparse
from tqdm import tqdm

import torch
import torchvision
from src.config import *
from src.dataset import get_test_loader
from src.trainer import compute_acc
from src.utils import *
from sklearn.metrics import classification_report


def test(args):
    test_loader = get_test_loader(args)
    model = get_model(args)
    model.load(os.path.join('checkpoint', args.checkpoint, 'weight', args.weight))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()
    total_num, correct = 0, 0
    test_bar = tqdm(test_loader, desc=f'Testing')
    pred_list = []
    label_list = []

    with torch.no_grad():
        for batch_data in test_bar:
            image, label = batch_data
            image = image.to(device)
            label = label.to(device)
            pred = model(image)
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
                        default='04-11-20-01-09',
                        help='weight path')
    parser.add_argument('--weight', type=str,
                        default='ep=0160-acc=0.8607.pth',
                        help='weight path')
    args = parser.parse_args()
    config = load_json(os.path.join('checkpoint', args.checkpoint, 'config.json'))
    args = argparse.Namespace(**vars(args), **config)
    test(args)
