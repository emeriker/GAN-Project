"""
[Module to generate Discriminator model]
"""
import tensorflow as tf

def get_model_discriminator(x_shape : int, y_shape : int) -> tf.keras.Model:
    """
    [Generates a discriminator model]

    Args:
        x_shape (int): [dimension x]
        y_shape (int): [dimension y]

    Returns:
        (tf.keras.Model): [Discrimator model]
    """
    input = tf.keras.Input(shape=(x_shape,y_shape,3), name='imput_layer')

    layer1 = tf.keras.layers.Conv2D(
        filters = 32,
        padding='same',
        kernel_size=3,
        activation='relu',
        name='layer1'
        )(input)
    layer2 = tf.keras.layers.Conv2D(
        filters = 32,
        padding='same',
        kernel_size=3,
        activation='relu',
        name='layer2'
        )(layer1)
    layer3 = tf.keras.layers.Add()([layer1,layer2])
    layer4 = tf.keras.layers.Conv2D(
        filters = 32,
        padding='same',
        kernel_size=3,
        activation='relu',
        name='layer4'
        )(layer3)
    layer5 = tf.keras.layers.Conv2D(
        filters = 32,
        padding='same',
        kernel_size=3,
        activation='relu',
        name='layer5'
        )(layer4)
    layer6 = tf.keras.layers.Add()([layer4,layer5])
    layer7 = tf.keras.layers.Add()([layer1,layer6])
    pool = tf.keras.layers.GlobalAveragePooling2D()(layer7)
    layer_fc = tf.keras.layers.Dense(16, activation='relu')(pool)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(layer_fc)

    model = tf.keras.Model(inputs=[input], outputs=[output])

    return model
