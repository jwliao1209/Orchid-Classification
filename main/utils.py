import os, datetime
import numpy as np
from tensorflow.keras.callbacks import Callback
from matplotlib import pyplot as plt

def get_current_time():
    today  = datetime.datetime.today()
    year   = today.year
    month  = today.month
    day    = today.day
    hour   = today.hour
    minute = today.minute
    second = today.second

    return f"{year}{month:02d}{day:02d}{hour:02d}{minute:02d}{second:02d}"


class ImageDisplayer(Callback):

    def __init__(self, generator, noise=None, img_root="result/generated_images"):
        self.generator = generator
        self.noise     = noise
        self.image_dir = os.path.join(img_root, get_current_time())

        os.makedirs(self.image_dir, exist_ok=True)

        if self.noise is None:
            self.noise = np.random.randn(16, generator.input_shape[1])

    def on_epoch_end(self, epoch, logs=None):
        images = self.generator.predict(self.noise)

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        plt.title(f"Epoch: {epoch:04d}")
        axes = axes.ravel()

        for ax, image in zip(axes, images):
            ax.imshow(image)
            ax.axis("off")
        
        plt.savefig(os.path.join(self.image_dir, f"epoch_{epoch:04d}.png"))
        plt.close()