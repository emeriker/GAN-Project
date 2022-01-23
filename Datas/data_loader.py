import os
import tensorflow as tf


def load_image_dir_dataset(
    data_dir : str, x_pixel : int, y_pixel : int
    ) -> tf.data.Dataset:
    """
    [Loads images from directory as batch dataset]

    Args:
        data_dir (str): [path as string]
        x_pixel (int): [nbr of pixels x]
        y_pixel (int): [nbr of pixels y]

    Returns:
        (tf.data.Dataset): [batch dataset from images dir]
    """
    train_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed = 123,
        image_size=(x_pixel, y_pixel))

    val_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed = 123,
        image_size=(x_pixel, y_pixel))

    return train_data, val_data

def create_noise_input(nbr_samples : int, x_dimension : int, y_dimension : int) -> tf.data.Dataset:
    """
    [Creates a random values of specified dimensions]

    Args:
        nbr_samples (int): []
        x (int): [description]
        y (int): [description]

    Returns:
        [tf.data.Dataset]: [description]
    """
    noise = tf.random.normal([nbr_samples,x_dimension,y_dimension,3])
    return noise

def load_mnist_data(batch_size : int):
    """
    [loads mnist dataset as batch]

    Args:
        batch_size (int): [batch size]

    Returns:
        (tf.data.Dataset): [mnist as dataset]
    """
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(6000).batch(batch_size)
    return train_dataset