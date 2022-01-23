"""
[Module to define the training process of models]
"""
import time
import tensorflow as tf
from Datas import data_loader
from Displayers import displayer
from IPython import display
from typing import Tuple


def run(
    model_gen: tf.keras.Model,
    model_disc: tf.keras.Model,
    dataset : tf.data.Dataset,
    epochs : int,
    batch_size : int,
    x_pixels : int,
    y_pixels : int
    ):
    """
    [Function to run training]

    Args:
        model_gen (tf.keras.Model): [Generator model]
        model_disc (tf.keras.Model): [Discrimiantor model]
        dataset (tf.data.Dataset): [Discrimiantor dataset]
        epochs (int): [nbr epochs]
        batch_size (int): [batch size]
        x_pixels (int): [x shape]
        y_pixels (int): [y shape]
    """
    train(
        model_gen,
        generator_loss,
        model_disc,
        discriminator_loss,
        dataset,
        epochs,
        batch_size,
        x_pixels,
        y_pixels
        )

def train(
    generator: tf.keras.Model,
    generator_loss: tf.keras.losses.BinaryCrossentropy,
    discriminator: tf.keras.Model,
    discriminator_loss: tf.keras.losses.BinaryCrossentropy,
    dataset : tf.data.Dataset,
    epochs : int,
    batch_size : int,
    x_pixels : int,
    y_pixels : int
    ) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    [Training process]

    Args:
        generator (tf.keras.Model): [Generator model]
        generator_loss ([type]): [loss generator]
        discriminator (tf.keras.Model): [Discrimiantor model]
        discriminator_loss ([type]): [loss discriminator]
        dataset (tf.data.Dataset): [Discrimiantor dataset]
        epochs (int): [nbr epochs]
        batch_size (int): [batch size]
        x_pixels (int): [x shape]
        y_pixels (int): [y shape]

    Returns:
        Tuple[tf.keras.Model, tf.keras.Model]: [description]
    """

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    for epoch in range(epochs):
        start = time.time()

        for image_batch, label_batch in dataset:
            train_step(
                generator,
                generator_loss,
                generator_optimizer,
                discriminator,
                discriminator_loss,
                discriminator_optimizer,
                cross_entropy,
                image_batch,
                batch_size,
                x_pixels,
                y_pixels
                )

        # Produce images for the GIF as you go
        display.clear_output(wait=True)

        noise = data_loader.create_noise_input(200, x_pixels, y_pixels)
        generated_images = generator(noise, training=False)
        generated_images = generated_images*255
        displayer.display_images_data_sample(generated_images)

        generated_images = generator(noise, training=False)
        generated_images = generated_images*255
        fake_output = discriminator(generated_images, training=False)

        gen_loss = generator_loss(cross_entropy, fake_output)

        print('-----> Loss generator :', gen_loss)

        noise2 = data_loader.create_noise_input(10, x_pixels, y_pixels)
        generated_images2 = generator(noise2, training=False)
        generated_images2 = generated_images2*255
        fake_output2 = discriminator(generated_images2, training=False)
        print('-----> 10 Fake disc pred : ', fake_output2)

        # Save the model every 15 epochs
        #if (epoch + 1) % 15 == 0:
        #    checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    return generator, discriminator

    # Generate after the final epoch
    #display.clear_output(wait=True)

@tf.function
def train_step(
    generator: tf.keras.Model,
    generator_loss: tf.keras.losses.BinaryCrossentropy,
    generator_optimizer: tf.keras.optimizers.Adam,
    discriminator:  tf.keras.Model,
    discriminator_loss: tf.keras.losses.BinaryCrossentropy,
    discriminator_optimizer: tf.keras.optimizers.Adam,
    cross_entropy: tf.keras.losses.BinaryCrossentropy,
    images: tf.data.Dataset,
    batch_size : int,
    x_pixels : int,
    y_pixels : int):
    """
    [Computes train]

    Args:
        generator (tf.keras.Model): [Generator model]
        generator_loss (tf.keras.losses.BinaryCrossentropy): [Generator loss function]
        generator_optimizer (tf.keras.optimizers.Adam): [Generator optimizer]
        discriminator (tf.keras.Model): [Discriminator model]
        discriminator_loss (tf.keras.losses.BinaryCrossentropy): [Discriminator loss function]
        discriminator_optimizer (tf.keras.optimizers.Adam): [Discriminator optimizer]
        cross_entropy (tf.keras.losses.BinaryCrossentropy): [tf.keras.losses.BinaryCrossentropy]
        images (tf.data.Dataset): [dataset of images]
        batch_size (int): [batch size]
        x_pixels (int): [x shape]
        y_pixels (int): [y shape]
    """
    noise = data_loader.create_noise_input(batch_size, x_pixels, y_pixels)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        generated_images = generator(noise, training=True)
        generated_images = generated_images*255
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(cross_entropy, fake_output)
        disc_loss = discriminator_loss(cross_entropy, real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def discriminator_loss(cross_entropy, real_output, fake_output):
    """[Discriminator loss function]

    Args:
        cross_entropy ([type]): [tf cross entropy]
        real_output ([type]): [Discriminator output of loss function of real images]
        fake_output ([type]): [Discriminator output of loss function of fake images]

    Returns:
        (tensorflow object): [tf cross entropy result]
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(cross_entropy, fake_output):
    """
    [Generator loss function]

    Args:
        cross_entropy ([type]): [tf cross entropy]
        fake_output ([type]): [Discriminator output of loss function of fake images]

    Returns:
        (tensorflow object): [tf cross entropy result]
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)