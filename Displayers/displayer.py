"""
[Module to display info on objects of project]
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def print_batch_data_info(data : tf.data.Dataset):
    """
    [Prints info of batch of a tf.data.Dataset]

    Args:
        data (tf.data.Dataset): [The batch dataset to print info]
    """
    for image_batch, labels_batch in data.take(1):
        print('Batch shape :', image_batch.shape)
        print('Batch label shape :', labels_batch.shape)

    print('Class names :', data.class_names)


def display_images_batch_data_sample(data : tf.data.Dataset):
    """
    [Displays sample images of a batch tf.data.Dataset]

    Args:
        data (tf.data.Dataset): [Batch tf.data.Dataset to display]
    """
    class_names = data.class_names

    for image_batch, labels_batch in data.take(1):
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            print(image_batch[i].dtype)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[labels_batch[i]])
            plt.axis("off")

def display_images_data_sample(data : tf.data.Dataset):
    """
    [Displays sample images of tf.data.Dataset]

    Args:
        data (tf.data.Dataset): [Batch tf.data.Dataset to display]
    """
    plt.figure(figsize=(10, 10))
    typeData = data.dtype
    print(typeData)
    i = 0
    for image_batch in data:
        if i == 4:
            break

        ax = plt.subplot(2, 2, i + 1)
        if (np.max(image_batch.numpy()) > 150):
            plt.imshow(image_batch.numpy().astype(np.float64)/255)
        else:
            plt.imshow(image_batch.numpy().astype(np.float64))
        plt.title('Generated image')
        plt.axis("off")
        
        i+=1
    plt.show()