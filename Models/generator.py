"""
[Module to generate Generator model]
"""
import tensorflow as tf

def get_model_generator(x_shape : int, y_shape : int) -> tf.keras.Model:
    """
    [Generates a Generator model]

    Args:
        x (int): [dimension x]
        y (int): [dimension y]

    Returns:
        (tf.keras.Model): [Generator model]
    """
    input = tf.keras.Input(shape=(x_shape, y_shape, 3), name='input_layer')#128,128,3))
    flat = tf.keras.layers.Flatten()(input)
    layernorm = tf.keras.layers.BatchNormalization()(flat)
    reshape = tf.keras.layers.Reshape((x_shape, y_shape, 3))(layernorm)

    layer1 = tf.keras.layers.Conv2D(
        filters = 32,
        padding='same',
        kernel_size=3,
        activation='relu',
        name='layer1'
        )(reshape)
    layer2 = tf.keras.layers.Conv2D(
        filters = 32,
        padding='same',
        kernel_size=3,
        activation='relu',
        name='layer2'
        )(layer1)
    #layer3 = tf.keras.layers.Add()([layer1,layer2])
    layer4 = tf.keras.layers.Conv2D(
        filters = 32,
        padding='same',
        kernel_size=3,
        activation='relu',
        name='layer4'
        )(layer2)
    layer5 = tf.keras.layers.Conv2D(
        filters = 32,
        padding='same',
        kernel_size=3,
        activation='relu',
        name='layer5'
        )(layer4)
    #layer6 = tf.keras.layers.Add()([layer4,layer5])
    layer_output = tf.keras.layers.Conv2D(
        filters = 3,
        padding='same',
        kernel_size=3,
        activation='sigmoid',
        name='layerOut'
        )(layer5)
    model = tf.keras.Model(inputs=[input], outputs=[layer_output])

    return model
