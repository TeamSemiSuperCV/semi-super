import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     GlobalMaxPooling2D, Input, Lambda)
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


# SE:
def gen_tsne(model, batch_t):
    # generates 2d static tse image
    # generates a file ./static/tsne.png
    labels, features = get_feats_for_tsne(model, batch_t)
    # RUN PCA
    pca = PCA(n_components=50)
    X = pca.fit_transform(features)
    # RUN TSNE
    #tsne =  TSNE(perplexity=79 ,n_components=2, metric='euclidean', random_state=2)
    tsne = TSNE(perplexity=20, n_components=2,
                metric='euclidean', random_state=2)
    tsne_result = tsne.fit_transform(X)

    # PLOT THE RESULTS
    sns.set(font_scale=1.5)
    sns.color_palette("colorblind")
    sns.set_style("white")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.scatterplot(x=tsne_result[:, 0],
                    y=tsne_result[:, 1], hue=labels, ax=ax, s=20)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.set_title('tSNE')
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.savefig('static/tsne.png')


def get_feats_for_tsne(model, batch_t):
    # gets the features from the dense layers
    # returns (labels, features, thisfeat)
    # load previous features
    features = npfile['feats']
    labels = npfile['labels']
    # add an extra label for this image (we want a it a different color)
    thislabel = np.max(labels)+1
    # get the output of the dense layer for this particular image
    layer = [model.get_layer('dense').output]
    model_d = Model(inputs=model.input, outputs=layer)
    thisfeat = model_d.predict(batch_t)
    # add this image's data to (labels, features)
    labels = np.append(labels, thislabel)
    features = np.concatenate((features, thisfeat), axis=0)
    return(labels, features)


def main():
    global model
    model = make_model(IMG_SIZE)
    model.build(IMG_SIZE)
    model.load_weights('FSL_ResNet50_XrayRemix.h5')
    model.trainable = False

    global gc
    gc = GradCam(model, 'lambda_1', 'jet_colors.npy')

    global npfile
    npfile = np.load('tsne_feats.npz')

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
