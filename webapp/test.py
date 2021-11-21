import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     GlobalMaxPooling2D, Lambda)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Model

from grad_cam import GradCam

IMG_SIZE = (256, 256, 3)


def img_preprocess(imgs):
    imgs = tf.image.resize(imgs, IMG_SIZE[:2], method='bilinear')
    return imgs


def model_preprocess(x):
    x = Rescaling(1/255)(x)
    return x


def make_model(input_shape):
    base_model = ResNet50(include_top=False, pooling=None, weights=None,
                          input_shape=input_shape)

    inputs = Input(shape=input_shape)
    x = model_preprocess(inputs)
    x = base_model(x)
    x = Lambda(lambda x: x, name='lambda_1')(x)
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def gen_heatmap(batch_t, img):
    superimp_g = gc.gen_grayscale_heatmap(batch_t, img, 0)
    superimp_c = gc.gen_color_heatmap(batch_t, img, 0)
    tf.io.write_file('static/saliency1.jpeg', tf.io.encode_jpeg(superimp_g))
    tf.io.write_file('static/saliency2.jpeg', tf.io.encode_jpeg(superimp_c))


def gen_tsne(model, batch_t):
    pass


def main():
    global model
    model = make_model(IMG_SIZE)
    model.build(IMG_SIZE)
    model.load_weights('FSL_ResNet50_XrayRemix.h5')
    model.trainable = False

    global gc
    gc = GradCam(model, 'lambda_1', 'jet_colors.npy')

    img_files = ['BACTERIA-7422-0001.jpeg', 'BACTERIA-30629-0001.jpeg',
                 'NORMAL-28501-0001.jpeg', 'NORMAL-32326-0001.jpeg']

    img_idx = 3
    img_fname = img_files[img_idx]
    # img_fname = 'test.jpeg'
    img = np.array(Image.open('static/' + img_fname))
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    batch = np.expand_dims(img, axis=0)
    batch_t = img_preprocess(batch)
    pred = model.predict(batch_t)
    pred_prob = f'{pred[0][0]:.3f}'
    print(f'{img_fname} ==> {pred_prob}')

    gen_heatmap(batch_t, img)
    gen_tsne(model, batch_t)


if __name__ == '__main__':
    main()
