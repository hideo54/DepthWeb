import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import flask
from google.cloud import storage
import uuid
import functions_framework
import traceback

import depth_estimation

def make_dataframes(lite=True):
    if lite:
        annotation_folder = '/val/'
        if not os.path.exists(os.path.abspath('.') + annotation_folder):
            annotation_zip = tf.keras.utils.get_file(
                'val.tar.gz',
                cache_subdir=os.path.abspath('.'),
                origin='http://diode-dataset.s3.amazonaws.com/val.tar.gz',
                extract=True,
            )

        filelist = []
        for root, dirs, files in os.walk(annotation_folder):
            for file in files:
                filelist.append(os.path.join(root, file))
        data = {
            'image': [x for x in filelist if x.endswith('.png')],
            'depth': [x for x in filelist if x.endswith('_depth.npy')],
            'mask': [x for x in filelist if x.endswith('_depth_mask.npy')],
        }
        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42)
        return df[:260].reset_index(drop='true'), df[260:].reset_index(drop='true')

    else:
        train_folder = '/train/'
        if not os.path.exists(os.path.abspath('.') + train_folder):
            train_zip = tf.keras.utils.get_file(
                'train.tar.gz',
                cache_subdir=os.path.abspath('.'),
                origin='http://diode-dataset.s3.amazonaws.com/train.tar.gz',
                extract=True,
            )

        val_folder = '/val/'
        if not os.path.exists(os.path.abspath('.') + val_folder):
            train_zip = tf.keras.utils.get_file(
                'val.tar.gz',
                cache_subdir=os.path.abspath('.'),
                origin='http://diode-dataset.s3.amazonaws.com/val.tar.gz',
                extract=True,
            )

        train_filelist = []
        for root, dirs, files in os.walk(train_folder):
            for file in files:
                train_filelist.append(os.path.join(root, file))
        train_data = {
            'image': [x for x in train_filelist if x.endswith('.png')],
            'depth': [x for x in train_filelist if x.endswith('_depth.npy')],
            'mask': [x for x in train_filelist if x.endswith('_depth_mask.npy')],
        }
        train_df = pd.DataFrame(train_data)

        val_filelist = []
        for root, dirs, files in os.walk(train_folder):
            for file in files:
                val_filelist.append(os.path.join(root, file))
        val_data = {
            'image': [x for x in val_filelist if x.endswith('.png')],
            'depth': [x for x in val_filelist if x.endswith('_depth.npy')],
            'mask': [x for x in val_filelist if x.endswith('_depth_mask.npy')],
        }
        val_df = pd.DataFrame(val_data)

        return train_df, val_df

def init():
    train_df, val_df = make_dataframes(lite=True)

    HEIGHT = 256
    WIDTH = 256
    LR = 0.0002
    EPOCHS = 30
    BATCH_SIZE = 32

    model_folder = '/model/'
    if not os.path.exists(os.path.abspath('.') + model_folder):
        print('Creating model...')
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LR,
            amsgrad=False,
        )
        model = depth_estimation.DepthEstimationModel()
        # Define the loss function
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )
        # Compile the model
        model.compile(optimizer, loss=cross_entropy)

        train_loader = depth_estimation.DataGenerator(
            data=train_df, batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH)
        )
        validation_loader = depth_estimation.DataGenerator(
            data=val_df, batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH)
        )
        model.fit(
            train_loader,
            epochs=EPOCHS,
            validation_data=validation_loader,
        )
        model.save('model')

if __name__ == '__main__':
    init()

def predict_depth(image_data):
    model = tf.keras.models.load_model('model')
    original_img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (256, 256))
    original_img = tf.image.convert_image_dtype(original_img, tf.float32)
    original_img = np.expand_dims(original_img, axis = 0)
    depth = model.predict(original_img).squeeze()
    return depth

def make_mini_depth_points(depth, original_image, length: int):
    original_image_size = np.shape(original_image)[:2] # [::-1] is not required here
    points = []
    for x in range(length):
        for y in range(length):
            original_x = np.round(x * original_image_size[0] / length).astype(int)
            original_y = np.round(y * original_image_size[1] / length).astype(int)
            xi = np.round(x * np.shape(depth)[0] / length).astype(int)
            yi = np.round(y * np.shape(depth)[1] / length).astype(int)
            original_color = original_image[original_x][original_y][::-1]
            original_color_hex = '#' + ''.join(map(lambda x: '0x{:02x}'.format(x)[2:], original_color.tolist()))
            points.append([original_x, original_y, depth[xi][yi], original_color_hex])
    return np.array(points)

def download_model(bucket):
    blobs = bucket.list_blobs(prefix='model/')
    for blob in blobs:
        os.makedirs(os.path.dirname(blob.name), exist_ok=True)
        blob.download_to_filename(blob.name)

@functions_framework.http
def make_predicted_image(request: flask.Request):
    headers = {
        'Access-Control-Allow-Origin': '*',
    }
    if request.method == 'POST' and request.files['file']:
        try:
            file = request.files['file']
            data = np.asarray(bytearray(file.read()), dtype=np.uint8)
            ext = file.filename.split('.')[-1]
            id = str(uuid.uuid4())
            generated_filename = f'{id}.png'

            client = storage.Client()
            bucket = client.get_bucket('depth-web')
            if os.path.exists(os.path.abspath('.') + '/model/'):
                download_model(bucket)

            original_image = cv2.imdecode(data, 1)
            original_image_size = np.shape(original_image)[:2][::-1]
            depth = predict_depth(original_image)
            depth_image = cv2.resize(
                np.expand_dims((depth * 255).astype(np.uint8), axis = 2), original_image_size
            )
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            depth_image_str = cv2.imencode(f'.png', depth_image)[1].tostring()
            bucket.blob(generated_filename).upload_from_string(depth_image_str, content_type='image/png')
            depth_points = make_mini_depth_points(depth, original_image, 100).tolist()
            res = {
                'filename': generated_filename,
                'depthPoints': depth_points,
            }
            return res, 200, headers
        except Exception as e:
            print(traceback.format_exc())
            return 'Error', 500, headers
    else:
        return 'No photo uploaded', 400, headers
