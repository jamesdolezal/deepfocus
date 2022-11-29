import click
import numpy
import matplotlib.pyplot as plt
import openslide
import tflearn
import numpy as np
import legacy_model
import os
import csv
from os.path import basename, join, exists
from skimage import filters, color
from typing import Callable, Tuple
from tqdm import tqdm


def analyze(
    slide_path: str,
    model: Callable,
    out: str = 'output',
    kernel_size: int = 64,
    buffer: int = 8,
    step_size: int = 1
) -> Tuple[int, int]:
    """Analyze a given image using DeepFocus.

    Args:
        buffer (int): Will load kernel_size x buffer."""

    # Initial preparation
    outputsVec = []
    buffersize = kernel_size * buffer
    k = kernel_size / step_size
    if not exists(out):
        os.makedirs(out)

    # Load slide & detect tissue via Otsu thresholding
    slide = openslide.open_slide(slide_path)
    thumbnail = np.array(slide.get_thumbnail((slide.level_dimensions[0][0] / kernel_size,
                                              slide.level_dimensions[0][1] / kernel_size)))
    gray_thumbnail = color.rgb2gray(thumbnail)
    val = filters.threshold_otsu(gray_thumbnail)
    tissue_mask = gray_thumbnail < max(val, 0.8)
    plt.imsave(join(out, 'tissue.png'), tissue_mask)
    focus_mask = np.zeros(thumbnail.shape[0:2], dtype=np.float32)
    if step_size > 1:
        new_shape = (int(focus_mask.shape[0] / step_size),
                     int(focus_mask.shape[1] / step_size))
        focus_mask = np.resize(focus_mask, new_shape)

    for i in tqdm(range(0, tissue_mask.shape[0] - buffer, buffer)): # Height
        for j in range(0, tissue_mask.shape[1] - buffer, buffer): # Width

            # Check for background; if so, skip
            if np.mean(tissue_mask[i:(i + buffer), j:(j + buffer)]) < (8/16):
                continue

            bigTile = numpy.array(slide.read_region((j * kernel_size, i * kernel_size), 0, [buffersize, buffersize]))
            bigTile = color.rgba2rgb(bigTile)
            sz = bigTile.itemsize
            h, w, c = bigTile.shape
            strided_shape = (int(h / k), int(w / k), kernel_size, kernel_size, c)
            strides = (step_size * kernel_size * sz * c * w,
                       step_size * kernel_size * sz * c,
                       sz * c * w,
                       sz * c,
                       sz)
            blocks = np.lib.stride_tricks.as_strided(bigTile, shape=strided_shape, strides=strides)
            blocks = blocks.reshape(blocks.shape[0] * blocks.shape[1],
                                    blocks.shape[2],
                                    blocks.shape[3],
                                    blocks.shape[4])
            predictions = model.predict(blocks)
            outputsVec.append(predictions)
            qwe = np.array(predictions)
            qwe = qwe.reshape(int(h / k), int(w / k), 2)
            focus_mask[int(i / step_size): int((i + buffer) / step_size),
                       int(j / step_size): int((j + buffer) / step_size)] = np.squeeze(qwe[:, :, 1])

    # Save results and export to CSV.
    plt.imsave(join(out, f'{basename(slide_path)}.png'), focus_mask, cmap=plt.cm.gray)
    with open(f'{basename(slide_path)}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(outputsVec)

    return focus_mask

@click.command()
@click.option('--slide', help='Path to slide', required=True, metavar='DIR')
def main(slide):

    # Load model
    tflearn.init_graph()
    g = legacy_model.deepfocus_v3()
    model = tflearn.DNN(g)
    model.load('checkpoints/ver5')

    # Generate results for a slide
    print("Working on: ", slide)
    results = analyze(slide, model)
    print("Results: ", results, results.shape)

if __name__ == "__main__":
    main()
