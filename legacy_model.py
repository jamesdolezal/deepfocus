
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.data_preprocessing import ImagePreprocessing
import tflearn.metrics
import tflearn.metrics


def deepfocus_v3(
    filters = (32, 32, 64, 128, 128),
    kernel_size = (5, 3, 3, 3, 3),
    pool = (0, 0, 1, 1, 1),
    fc = (128, 64),
    training = False
):
    assert len(filters) == len(kernel_size) == len(pool)
    img_prep = ImagePreprocessing()
    img_prep.add_samplewise_zero_center()

    if training:
        imgaug = tflearn.ImageAugmentation()
        imgaug.add_random_flip_leftright()
        imgaug.add_random_flip_updown()
        imgaug.add_random_90degrees_rotation()
    else:
        imgaug = None

    # Input
    network = input_data(shape=[None, 64, 64, 3], data_preprocessing=img_prep, data_augmentation=imgaug, name='input')

    # Convolutional layers
    for idx in range(len(filters)):
        network = conv_2d(network, filters[idx], kernel_size[idx], activation='relu', regularizer="L2")  # padding same
        network = batch_normalization(network)
        if pool[idx]:
            network = max_pool_2d(network, 2)

    # Fully connected layers
    for idx in range(len(fc)):
        network = fully_connected(network, fc[idx], activation='relu', regularizer="L2")
        network = batch_normalization(network)
        network = dropout(network, 0.2)

    g = tflearn.fully_connected(network, 2, activation='softmax')
    g = tflearn.regression(g, optimizer='SGD', loss='categorical_crossentropy', metric='default' , learning_rate=0.01, batch_size=64)
    return g
