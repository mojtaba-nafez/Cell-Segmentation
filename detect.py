# Copyright 2021 The AI-Medic\Cell-Segmentation Authors. All Rights Reserved.
# License stuff will be written here later...

"""
This module is to call model and give it the input image; then get the output image(s)
    from it and pass the output image to the streamlit ui.
"""

from models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
import glob

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Detect:
    """
    The module to get output image from the model and pass it to the streamlit ui.
    This module has below functions:
        -   detect
        -   detect_from_path
    """
    def __init__(self, model_name, weight_path, **kwargs):
        """
        Parameters
        ----------
        model_name: str ; name of the model.
        weight_path: str ; path to the corresponding model's weights.
        **kwargs

        Returns
        -------
        None
        """
        self.model = load_model(model_name=model_name, **kwargs)
        self.model.load_weights(weight_path)

    def detect(self, img):
        """
        This function reads the image array and apply some preprocessing (if needeed)
            on it and pass it to model; then provide the model's output image(s).
        
        Parameters
        ----------
        img: input image array.

        Returns
        -------
        result: array ; output image of the model.
        """
        # apply necessary preprocessing
        # this preprocessing works just for unet model
        img = np.array(img)
        img_s = (np.array(img.shape) >> 4) << 4
        img = np.array(Image.fromarray(img).convert("RGB"), dtype=np.float32) / 255.
        img = img[np.newaxis, :img_s[0], :img_s[1], :]

        # predict
        result = self.model.predict(img)

        # apply necessary post-processing
        result = np.array(result * 255, dtype=np.uint8).squeeze()

        # return the results
        return result

    def detect_from_path(self, img_path):
        """
        This function reads the image file and apply some preprocessing (if needeed)
            on it and pass it to model; then provide the model's output image(s).
        
        Parameters
        ----------
        img_path: str ; path to input image file.

        Returns
        -------
        array ; output image of the model.
        """
        # make necessary modifications
        img = Image.open(img_path)
        return self.detect(img)


if __name__ == '__main__':
    detect = Detect(
        'unet',                         # saved model name
        './weights/unet/weights',       # path to saved model weights
        input_shape=(None, None, 3)     # input shape of saved model
    )
    results = detect.detect_from_path(
        img_path=glob.glob("./streamlit/files/random-images/*")[0])
    print(results)
