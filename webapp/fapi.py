from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras import Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     GlobalMaxPooling2D, Lambda)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Model

from grad_cam import GradCam

IMG_SIZE = (256, 256, 3)

img_fnames = {
    '0': "BACTERIA-7422-0001.jpeg",
    '1': "NORMAL-28501-0001.jpeg",
    '2': "BACTERIA-30629-0001.jpeg",
    '3': "NORMAL-32326-0001.jpeg",
}

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount('/static', StaticFiles(directory='static'), name='static')


@app.get('/', response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/diagnose/{idx}', response_class=HTMLResponse)
async def read_item(request: Request, idx: str):
    img_fname = img_fnames[idx]
    return img_predict(img_fname, request)


@app.post('/upload', response_class=HTMLResponse)
async def form(request: Request, file: UploadFile = File(...)):
    print(file.filename)
    # contents = await file.read()
    with open('static/upload.jpeg', 'wb') as out_file:
        out_file.write(await file.read())
    return img_predict('upload.jpeg', request)


def gen_heatmaps(batch_t, img):
    superimp_ssl = gc_ssl.gen_color_heatmap(batch_t, img, 0)  # SSL Model
    superimp_fsl = gc_fsl.gen_grayscale_heatmap(batch_t, img, 0)  # FSL Model
    tf.io.write_file('static/saliency2.jpeg', tf.io.encode_jpeg(superimp_ssl))
    tf.io.write_file('static/saliency1.jpeg', tf.io.encode_jpeg(superimp_fsl))


def gen_tsne(featlayer, batch_t):
    # generates 2d static tse image
    # generates a file ./static/tsne.png
    labels, features = get_feats_for_tsne(featlayer, batch_t)
    # RUN PCA
    pca = PCA(n_components=50)
    X = pca.fit_transform(features)
    # RUN TSNE
    #tsne =  TSNE(perplexity=79 ,n_components=2, metric='euclidean', random_state=2)
    tsne = TSNE(perplexity=50, n_components=2,
                metric='euclidean', random_state=2)
    tsne_result = tsne.fit_transform(X)

        # PLOT THE RESULTS
    sns.set(font_scale = 1.5)
    sns.set_style("white")
    fig,ax = plt.subplots(1,1,figsize=(8,8));
    dotsize = (labels==2)+1.
    log_0 = labels == 0
    log_1 = labels == 1
    log_2 = labels == 2
    plt.scatter(x=tsne_result[log_0,0], y=tsne_result[log_0,1],
                c= '#30a2da', #c='#e5ae38', #c= '#fc4f30', #c='gray', #c='tomato',
                s=20, alpha=0.5, label='Normal')
    plt.scatter(x=tsne_result[log_1,0], y=tsne_result[log_1,1],
                c= '#fc4f30', #c='#6d904f', #c= '#30a2da', #c='red', #c='royalblue',
                s=20, alpha=0.5, label='Pneumonia')
    plt.scatter(x=tsne_result[log_2,0], y=tsne_result[log_2,1],
                c='black',
                s=200, alpha=1,marker='*', label='Diagnosed')

    # sns.scatterplot(x=tsne_result[:,0], y=tsne_result[:,1], hue=labels, ax=ax,size=dotsize,alpha=.8, palette="colorblind")
    sns.despine()
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.set_title('tSNE')
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0.0, frameon=False);
    ax.legend(loc='best', borderaxespad=0.0, frameon=False, labelspacing=0.25,handletextpad=-.25);
    plt.savefig('static/tsne.png')


def get_feats_for_tsne(featlayer, batch_t):
    # gets the features from the dense layers
    # returns (labels, features, thisfeat)
    # load previous features
    features = npfile['feats']
    labels = npfile['labels']
    # add an extra label for this image (we want a it a different color)
    thislabel = np.max(labels)+1
    # get the output of the dense layer for this particular image
    thisfeat = featlayer.predict(batch_t)
    # add this image's data to (labels, features)
    labels = np.append(labels, thislabel)
    features = np.concatenate((features, thisfeat), axis=0)
    return(labels, features)


def img_predict(img_fname, request):
    img = np.array(Image.open('static/' + img_fname))
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    batch = np.expand_dims(img, axis=0)
    batch_t = img_preprocess(batch)
    pred = model_fsl.predict(batch_t)
    pred_prob = f'{pred[0][0]:.3f}'
    print(f'{img_fname} ==> {pred_prob}')

    gen_heatmaps(batch_t, img)
    gen_tsne(featLayer_fsl, batch_t)

    rand_refresh = str(int(time() % 8192))
    return templates.TemplateResponse("diagnose.html",
                                      {"request": request,
                                       "probability": pred_prob,
                                       "f_example": f'{img_fname}?{rand_refresh}',
                                       "f_tsne": f'tsne.png?{rand_refresh}',
                                       "f_saliency1": f'saliency1.jpeg?{rand_refresh}',
                                       "f_saliency2": f'saliency2.jpeg?{rand_refresh}',
                                       })


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


def main():
    # fully supervised model
    global model_fsl
    model_fsl = make_model(IMG_SIZE)
    model_fsl.build(IMG_SIZE)
    model_fsl.load_weights('FSL_ResNet50_XrayRemix.h5')
    model_fsl.trainable = False

    # modeloutput of feature layers (model_fsl)
    global featLayer_fsl
    layer = [model_fsl.get_layer('dense').output]
    featLayer_fsl = Model(inputs=model_fsl.input, outputs=layer)



    global npfile
    npfile = np.load('tsne_feats.npz')

    global gc_fsl, gc_ssl
    gc_fsl = GradCam(model_fsl, 'lambda_1', 'jet_colors.npy')  # FSL
    gc_ssl = GradCam(model_fsl, 'lambda_1', 'jet_colors.npy')  # SSL


if __name__ == '__main__':
    main()
    uvicorn.run(app, host="0.0.0.0", port=8080)
else:
    main()
