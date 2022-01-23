"""
[main]
"""
from gan import GAN
from Models import discriminiator, generator
from Datas import data_loader


def workflow():
    """
    [workflow]
    """
    epochs = 5000
    x_pixels = 32
    y_pixels = 32
    num_examples_to_generate = 16
    seed = 2022
    batch_size = 32

    train_dataset, val_dataset = data_loader.load_image_dir_dataset(
        "C:/Users/emeri/data/DogImages", x_pixels, y_pixels
        )

    model_gen = generator.get_model_generator(x_pixels, y_pixels)
    model_disc = discriminiator.get_model_discriminator(x_pixels, y_pixels)

    gan = GAN()

    gan.set_training_hyper_params(
        epochs, num_examples_to_generate, x_pixels, y_pixels, batch_size, seed
        )

    gan.set_images_dataset(train_dataset, val_dataset)

    gan.set_discriminator_model(model_disc)
    gan.set_generator_model(model_gen)

    gan.train()
