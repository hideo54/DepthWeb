import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt

import depth_estimation

def init():
    annotation_folder = "/val/"
    if not os.path.exists(os.path.abspath(".") + annotation_folder):
        annotation_zip = tf.keras.utils.get_file(
            "val.tar.gz",
            cache_subdir=os.path.abspath("."),
            origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
            extract=True,
        )

    path = "val/indoors"
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))

    filelist.sort()
    data = {
        "image": [x for x in filelist if x.endswith(".png")],
        "depth": [x for x in filelist if x.endswith("_depth.npy")],
        "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
    }
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42)

    HEIGHT = 256
    WIDTH = 256
    LR = 0.0002
    EPOCHS = 30
    BATCH_SIZE = 32

    model_folder = "/model/"
    if not os.path.exists(os.path.abspath(".") + model_folder):
        print("Creating model...")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LR,
            amsgrad=False,
        )
        model = depth_estimation.DepthEstimationModel()
        # Define the loss function
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        # Compile the model
        model.compile(optimizer, loss=cross_entropy)

        train_loader = depth_estimation.DataGenerator(
            data=df[:260].reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH)
        )
        validation_loader = depth_estimation.DataGenerator(
            data=df[260:].reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH)
        )
        model.fit(
            train_loader,
            epochs=EPOCHS,
            validation_data=validation_loader,
        )
        model.save("model")

def main():
    assert os.path.exists(os.path.abspath(".") + "/model/")
    model = tf.keras.models.load_model('model')
    image_path = 'sample.jpg'
    original_img = cv2.imread(image_path)
    original_img_size = np.shape(original_img)[:2][::-1]
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (256, 256))
    original_img = tf.image.convert_image_dtype(original_img, tf.float32)
    original_img = np.expand_dims(original_img, axis = 0)

    pred = model.predict(original_img)
    pred = pred.squeeze()
    pred = np.expand_dims(pred, axis = 2)
    pred = cv2.resize(pred, original_img_size)
    plt.imshow(pred)
    plt.show()
    # image.save_img('predicted.jpg', pred)

init()
main()
