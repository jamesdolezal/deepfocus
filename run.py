import click
import numpy as np
import slideflow as sf
from PIL import Image
from tqdm import tqdm
from rich import print as print


def check_focus_legacy(slide, buffer, kernel, qc, mag):
    """Check for in-focus areas using the legacy TFLearn model."""

    import tflearn
    from legacy_model import deepfocus_v3

    full_size = kernel * buffer
    wsi = sf.WSI(slide, tile_px=(kernel * buffer), tile_um=mag)
    wsi.qc(qc)

    # Initialize & load model
    tflearn.init_graph()
    model = tflearn.DNN(deepfocus_v3())
    model.load('checkpoints/ver5')

    # Initialize whole-slide reader.
    generator = wsi.build_generator(shuffle=False, show_progress=False, img_format='numpy', num_processes=48)
    dts = generator()

    # Generate prediction for slide.
    focus_mask = np.ones((wsi.grid.shape[1] * buffer, wsi.grid.shape[0] * buffer), dtype=np.float32)
    for item in tqdm(dts, desc="Generating...", total=wsi.estimated_num_tiles):
        img = np.clip(item['image'].astype(np.float32) / 255, 0, 1)
        sz = img.itemsize
        grid_i = item['grid'][1]
        grid_j = item['grid'][0]
        batch = np.lib.stride_tricks.as_strided(img,
                                                shape=(buffer, buffer, kernel, kernel, 3),
                                                strides=(kernel * sz * 3 * full_size,
                                                         kernel * sz * 3,
                                                         sz * 3 * full_size,
                                                         sz * 3,
                                                         sz))
        batch = batch.reshape(batch.shape[0] * batch.shape[1],
                              batch.shape[2],
                              batch.shape[3],
                              batch.shape[4])
        y_pred = model.predict(batch)[:, 1]

        predictions = y_pred.reshape(buffer, buffer)
        focus_mask[grid_i * buffer: grid_i * buffer + buffer, grid_j * buffer: grid_j * buffer + buffer] = predictions

    # Show results.
    Image.fromarray((focus_mask * 255).astype(np.uint8)).show()
    return focus_mask

# -----------------------------------------------------------------------------

@click.command()
@click.option('--slide', help='Path to slide', required=True, metavar='DIR')
@click.option('--buffer', help='Path to slide', metavar=int, default=8)
@click.option('--kernel', help='Kernel size for prediction.', metavar=int, default=64)
@click.option('--mag', help='Magnification level. Defaults to 40x.', metavar=str, default='40x')
@click.option('--qc', help="Base QC method to use.", metavar=str, default='otsu')
@click.option('--legacy', help="Use legacy TFLearn API.", metavar=bool, is_flag=True, default=False)
def check_focus(slide, buffer, kernel, mag, qc, legacy):
    """Check for in-focus areas."""

    if legacy:
        return check_focus_legacy(slide, buffer, kernel, mag)

    from keras_model import deepfocus_v3, load_checkpoint

    full_size = kernel * buffer
    wsi = sf.WSI(slide, tile_px=(kernel * buffer), tile_um=mag)
    wsi.qc(qc)

    # Initialize & load model
    model = deepfocus_v3()
    load_checkpoint(model, 'checkpoints/ver5')

    # Initialize whole-slide reader.
    generator = wsi.build_generator(shuffle=False, show_progress=False, img_format='numpy', num_processes=24)
    dts = generator()

    # Generate prediction for slide.
    focus_mask = np.ones((wsi.grid.shape[1] * buffer, wsi.grid.shape[0] * buffer), dtype=np.float32)
    for item in tqdm(dts, desc="Generating...", total=wsi.estimated_num_tiles):
        img = np.clip(item['image'].astype(np.float32) / 255, 0, 1)
        sz = img.itemsize
        grid_i = item['grid'][1]
        grid_j = item['grid'][0]
        batch = np.lib.stride_tricks.as_strided(img,
                                                shape=(buffer, buffer, kernel, kernel, 3),
                                                strides=(kernel * sz * 3 * full_size,
                                                         kernel * sz * 3,
                                                         sz * 3 * full_size,
                                                         sz * 3,
                                                         sz))
        batch = batch.reshape(batch.shape[0] * batch.shape[1], *batch.shape[2:])
        y_pred = model(batch)[:, 1].numpy()
        predictions = y_pred.reshape(buffer, buffer)
        focus_mask[grid_i * buffer: grid_i * buffer + buffer, grid_j * buffer: grid_j * buffer + buffer] = predictions

    # Show results.
    Image.fromarray((focus_mask * 255).astype(np.uint8)).show()

    # Sample use as QC mask.
    wsi = sf.WSI(slide, tile_px=299, tile_um=302)
    wsi.apply_qc_mask(focus_mask > 0.5)
    wsi.preview().show()

    return focus_mask

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    check_focus()