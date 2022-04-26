import tensorflow as tf
from tensorflow.keras import layers, Model, Input, Sequential

def get_discriminator(input_shape=(224, 224, 3)):
    inputs = Input(input_shape)

    x = DoubleConv2D(32, 3, strides=2, padding="same")(inputs)
    x = DoubleConv2D(64, 3, strides=2, padding="same")(x)
    x = DoubleConv2D(128, 3, strides=2, padding="same")(x)
    x = DoubleConv2D(256, 3, strides=2, padding="same")(x)
    x = DoubleConv2D(512, 3, strides=2, padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(1)(x)

    return Model(inputs, outputs)

def get_generator(input_shape=(128,)):
    inputs = Input(128)

    x = FCBlock(7*7*512, "relu")(inputs)
    x = layers.Reshape([7, 7, 512])(x)
    x = DoubleConv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = DoubleConv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = DoubleConv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = DoubleConv2DTranspose(32, 3, strides=2, padding="same")(x)

    outputs = layers.Conv2DTranspose(3, 3, strides=2, padding="same", activation="sigmoid")(x)

    return Model(inputs, outputs)


class DoubleConv2D(Sequential):

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        activation="relu"
    ):
        super().__init__([
            layers.Conv2D(filters//2, kernel_size, padding="same"),
            layers.LayerNormalization(),
            layers.Conv2D(filters, kernel_size, strides=strides, padding=padding),
            layers.LayerNormalization(),
            layers.PReLU()
        ])


class DoubleConv2DTranspose(Sequential):

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        activation="relu"
    ):
        super().__init__([
            layers.Conv2DTranspose(filters//2, kernel_size, padding="same"),
            layers.LayerNormalization(),
            layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding),
            layers.LayerNormalization(),
            layers.PReLU()
        ])


class FCBlock(Sequential):

    def __init__(self, units, activation="relu"):
        super().__init__([
            layers.Dense(units),
            layers.LayerNormalization(),
            layers.PReLU()
        ])


class GAN(Model):

    def __init__(
        self,
        generator,
        discriminator
    ):
        super(GAN, self).__init__()
        self.generator     = generator
        self.discriminator = discriminator
        self.latent_dim    = generator.input_shape[1]

    def compile(self, d_optimizer, g_optimizer, loss):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss = loss

    def call(self, data):
        batch_size  = tf.shape(data)[0]
        noise       = tf.random.normal(shape=(batch_size, self.latent_dim))

        return self.generator(noise)

    def compute_discriminator_loss(self, real_prob, fake_prob):
        real_loss = self.cross_entropy(tf.ones_like(real_prob), real_prob)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_prob), fake_prob)

        return real_loss + fake_loss

    def compute_generator_loss(self, fake_prob):
        return self.cross_entropy(tf.ones_like(fake_prob), fake_prob)

    def train_step(self, real_images):
        batch_size  = tf.shape(real_images)[0]
        noise       = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_images = self.generator(noise, training=False)
        all_images  = tf.concat([real_images, fake_images], axis=0)
        all_labels  = tf.concat(
            [tf.ones((batch_size, 1)) - 0.1, 0.1 + tf.zeros((batch_size, 1))],
            axis=0
        )

        with tf.GradientTape() as tape:
            all_probs = self.discriminator(all_images, training=True)
            disc_loss = self.loss(all_labels, all_probs)

        grad_disc = tape.gradient(
            disc_loss,
            self.discriminator.trainable_variables
        )
        self.d_optimizer.apply_gradients(
            zip(grad_disc, self.discriminator.trainable_variables)
        )

        # noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_probs  = self.discriminator(fake_images, training=False)
            gen_loss    = self.loss(tf.ones((batch_size, 1)), fake_probs)

        grad_gen = tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )
        self.g_optimizer.apply_gradients(
            zip(grad_gen, self.generator.trainable_variables)
        )

        return {"disc_loss": disc_loss, "gen_loss": gen_loss}