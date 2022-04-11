import torch
import random


class RandomNoise():
    ''' Random noise from Gaussian distribution'''
    def __init__(self, sig=0.005, p=0.1):
        self.sig = sig
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            image += self.sig * torch.randn(image.shape)

        return image


class GridMask(object):
    def __init__(self, shape=(32, 32), dmin=5, dmax=10, ratio=0.7, p=0.3):
        self.shape = shape
        self.dmin = dmin
        self.dmax = dmax
        self.ratio = ratio
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        d = random.randint(self.dmin, self.dmax)
        dx, dy = random.randint(0, d-1), random.randint(0, d-1)
        sl = int(d * (1-self.ratio))
        for i in range(dx, self.shape[0], d):
            for j in range(dy, self.shape[1], d):
                row_end = min(i+sl, self.shape[0])
                col_end = min(j+sl, self.shape[1])
                img[:, i:row_end, j:col_end] = 0
        return img


if __name__ == '__main__':
    pass
    
