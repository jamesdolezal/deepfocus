import tensorflow as tf
from tensorflow.keras.regularizers import L2


def transform_variable_name(name):
    """Transform a variable name from new (Keras) to old (TFLearn) format."""
    if name.endswith('kernel'):
        return name.split('kernel')[0] + 'W'
    if name.endswith('bias'):
        return name.split('bias')[0] + 'b'
    return name


def deepfocus_v3(
    filters = (32, 32, 64, 128, 128),
    kernel_size = (5, 3, 3, 3, 3),
    fc = (128, 64)
):
    assert len(filters) == len(kernel_size)

    # Input.
    inp = tf.keras.layers.InputLayer(input_shape=(64, 64, 3), name='input')

    # Pre-processing.
    x = inp.output - tf.math.reduce_mean(inp.output, keepdims=True)

    # Convolutional layers.
    x = tf.keras.layers.Conv2D(filters[0], kernel_size[0], activation='relu', kernel_regularizer=L2, padding='same', name='Conv2D')(x)
    x = tf.keras.layers.BatchNormalization(name='BatchNormalization')(x)

    x = tf.keras.layers.Conv2D(filters[1], kernel_size[1], activation='relu', kernel_regularizer=L2, padding='same', name='Conv2D_1')(x)
    x = tf.keras.layers.BatchNormalization(name='BatchNormalization_1')(x)

    x = tf.keras.layers.Conv2D(filters[2], kernel_size[2], activation='relu', kernel_regularizer=L2, padding='same', name='Conv2D_2')(x)
    x = tf.keras.layers.BatchNormalization(name='BatchNormalization_2')(x)
    x = tf.keras.layers.MaxPool2D((2,2), padding='same')(x)

    x = tf.keras.layers.Conv2D(filters[3], kernel_size[3], activation='relu', kernel_regularizer=L2, padding='same', name='Conv2D_3')(x)
    x = tf.keras.layers.BatchNormalization(name='BatchNormalization_3')(x)
    x = tf.keras.layers.MaxPool2D((2,2), padding='same')(x)

    x = tf.keras.layers.Conv2D(filters[4], kernel_size[4], activation='relu', kernel_regularizer=L2, padding='same', name='Conv2D_4')(x)
    x = tf.keras.layers.BatchNormalization(name='BatchNormalization_4')(x)
    x = tf.keras.layers.MaxPool2D((2,2), padding='same')(x)

    # Fully connected layers.
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(fc[0], activation='relu', kernel_regularizer=L2, name='FullyConnected')(x)
    x = tf.keras.layers.BatchNormalization(name='BatchNormalization_5')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(fc[1], activation='relu', kernel_regularizer=L2, name='FullyConnected_1')(x)
    x = tf.keras.layers.BatchNormalization(name='BatchNormalization_6')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(2, activation='softmax', name='FullyConnected_2')(x)
    model = tf.keras.Model(inputs=inp.input, outputs=x)
    return model

def load_checkpoint(model, ckpt, verbose=False):
    reader = tf.compat.v1.train.load_checkpoint(ckpt)
    ckpt_vars = tf.compat.v1.train.list_variables(ckpt)
    ckpt_layers = list(set([c[0].split('/')[0] for c in ckpt_vars]))
    print("Loading {} variables ({} layers) from checkpoint {}.".format(len(ckpt_vars), len(ckpt_layers), ckpt))

    if verbose:
        for name, tensor_shape in ckpt_vars:
            print("\t", name, tensor_shape)

    for layer_name in ckpt_layers:
        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            if verbose:
                print("Skipping layer {}".format(layer_name))
        else:
            if verbose:
                print("Working on layer {}".format(layer_name))
            ckpt_vals = []
            for varname in [w.name for w in layer.weights]:
                varname = varname.split(':0')[0]
                varname = transform_variable_name(varname)
                ckpt_vals.append(reader.get_tensor(varname))
                if verbose:
                    print("\tWorking on varname", varname, ckpt_vals[-1].shape)
            layer.set_weights(ckpt_vals)
            if verbose:
                print("\tSet {} variables to layer {}".format(len(layer.weights), layer_name))
    return model