import argparse, json

import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from main.gan_module import get_generator, get_discriminator, GAN
from main.utils import ImageDisplayer


def main(epochs, learning_rate, batch_size):
    train   = pd.read_csv("data/train.csv", names=["filename", "class"])
    dataset = ImageDataGenerator(rescale=1/255).flow_from_dataframe(
        train,
        "data/image",
        target_size=(224, 224),
        class_mode=None,
        batch_size=batch_size
    )

    generator     = get_generator()
    discriminator = get_discriminator()

    gan = GAN(generator, discriminator)
    gan.compile(
        d_optimizer=tf.keras.optimizers.Adam(learning_rate),
        g_optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    )
    gan.fit(dataset, epochs=epochs, callbacks=[ImageDisplayer(generator)])
    generator.save_weights("result/generator_weights.hdf5")
    discriminator.save_weights("result/discriminator_weights.hdf5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=10000)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-b", "--batch_size", type=int, default=32)

    args = vars(parser.parse_args())

    json.dump(args, open("result/gan_params.json", "w"), indent=4)
    main(**args)
