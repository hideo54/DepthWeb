import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
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
    if not os.path.exists(os.path.abspath(".") + annotation_folder):
        print("Creating model...")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LR,
            amsgrad=False,
        )
        model = DepthEstimationModel()
        # Define the loss function
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        # Compile the model
        model.compile(optimizer, loss=cross_entropy)

        train_loader = DataGenerator(
            data=df[:260].reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH)
        )
        validation_loader = DataGenerator(
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
    img = image.load_img('IMG_2943.jpeg', target_size = (256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    pred = model.predict(img)
    pred = pred.squeeze()
    pred = np.expand_dims(pred, axis = 2)
    print(pred.shape)
    image.save_img('predicted.jpg', pred)

main()
