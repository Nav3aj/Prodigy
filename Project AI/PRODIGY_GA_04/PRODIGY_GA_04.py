"""pix2pix_cgan.py
Instruction: Image-to-Image translation using Pix2Pix cGAN (TensorFlow).
"""

# Topic: Imports
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import argparse
import os

# Topic: Load and preprocess dataset
def load_dataset(batch_size=1, img_size=256):
    dataset, info = tf.keras.datasets.mnist.load_data()
    # Placeholder: in practice, use a paired dataset like facades
    # Here we just convert MNIST to 3-channel and duplicate
    train = tf.expand_dims(dataset[0], -1)
    train = tf.image.grayscale_to_rgb(tf.image.resize(train[...,None], [img_size, img_size]))
    train_ds = tf.data.Dataset.from_tensor_slices((train, train))
    train_ds = train_ds.batch(batch_size)
    return train_ds

# Topic: Build model
def build_generator():
    return pix2pix.unet_generator(3, norm_type='instancenorm')

def build_discriminator():
    return pix2pix.discriminator()

# Topic: Training loop
def train(epochs=1, batch_size=1):
    train_ds = load_dataset(batch_size)
    generator = build_generator()
    discriminator = build_discriminator()
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    for epoch in range(epochs):
        for input_image, target in train_ds:
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_output = generator(input_image, training=True)
                disc_real = discriminator([input_image, target], training=True)
                disc_fake = discriminator([input_image, gen_output], training=True)
                gen_loss = pix2pix.generator_loss(disc_fake, gen_output, target)
                disc_loss = pix2pix.discriminator_loss(disc_real, disc_fake)
            grads_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
            grads_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            generator_optimizer.apply_gradients(zip(grads_gen, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_variables))
        print(f"Epoch {epoch+1}/{epochs} | Gen Loss: {gen_loss.numpy():.4f} | Disc Loss: {disc_loss.numpy():.4f}")

# Topic: CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pix2Pix cGAN training")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch)
