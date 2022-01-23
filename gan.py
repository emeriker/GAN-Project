import tensorflow as tf
from Train import train

class GAN():
    def __init__(self) -> None:
        #Training hyper params
        self.epochs : int = None
        self.num_examples_to_generate : int = None
        self.batch_size : int = None
        self.seed : int = None
        self.x_image : int = None
        self.y_image : int = None

        self.train_dataset : tf.data.Dataset = None
        self.val_dataset : tf.data.Dataset = None

        self.model_disc : tf.keras.Model = None
        self.model_gen : tf.keras.Model = None

    def set_training_hyper_params(self, epochs : int, num_examples_to_generate : int, x_image : int, y_image : int, batch_size : int, seed : int):
        self.epochs = epochs
        self.x_image = x_image
        self.y_image = y_image
        self.num_examples_to_generate = num_examples_to_generate
        self.batch_size = batch_size
        self.seed = seed

    def set_images_dataset(self, train_dataset : tf.data.Dataset, val_dataset : tf.data.Dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def set_discriminator_model(self, model_disc : tf.keras.Model):
        self.model_disc = model_disc

    def set_generator_model(self, model_gen : tf.keras.Model):
        self.model_gen = model_gen

    def train(self):

        #Train
        self.model_gen, self.model_disc = train.run(
            self.model_gen,
            self.model_disc,
            self.train_dataset,
            self.epochs,
            self.batch_size,
            self.x_image,
            self.y_image
        )

        return self.model_gen, self.model_disc



