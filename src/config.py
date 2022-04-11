import torch

from src.model import *
from src.losses import FL, MCCE
from src.optimizer import ranger21


def get_model(args):
    Model = {
        'EfficientB4':   EfficientNet_b4,
        'MCEfficientB4': MCEfficientNet_b4,
    }
    model = Model[args.model](args.num_classes)

    return model


def get_criterion(args, device):
    Losses = {
        'CE':   nn.CrossEntropyLoss,
        'MCCE': MCCE.MCCE_Loss,
        'FL':   FL.FocalLoss,
        'FLSD': FL.FocalLossAdaptive
    }
    criterion = Losses[args.loss]()

    return criterion


def get_optimizer(args, model):
    Optimizer = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'Ranger': ranger21.Ranger21
    }
    optimizer = Optimizer[args.optim](
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    return optimizer


def get_scheduler(args, optimizer):
    Scheduler = {
        'step': torch.optim.lr_scheduler.StepLR(
              optimizer=optimizer,
              step_size=args.step_size,
              gamma=args.gamma
        ),
        'cos': torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.epoch
        )
    }
    scheduler = Scheduler[args.scheduler]

    return scheduler
