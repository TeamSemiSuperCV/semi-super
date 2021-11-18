from time import time

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from tensorflow.keras import Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     GlobalMaxPooling2D, Lambda)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Model

from grad_cam import GradCam
from tsne import TSNE

IMG_SIZE_FSL = (256, 256, 3)
IMG_SIZE_SSL = (224, 224, 3)

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

@app.get('/about.html', response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get('/team.html', response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("team.html", {"request": request})


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


def gen_heatmap_fsl(batch_t, img):
    superimp_fsl = gc_fsl.gen_color_heatmap(batch_t, img, 0)  # FSL Model
    tf.io.write_file('static/saliency1.jpeg', tf.io.encode_jpeg(superimp_fsl))


def gen_heatmap_ssl(batch_t, img):
    superimp_ssl = gc_ssl.gen_color_heatmap(batch_t, img, 1)  # SSL Model
    tf.io.write_file('static/saliency2.jpeg', tf.io.encode_jpeg(superimp_ssl))


def img_predict(img_fname, request):
    img = np.array(Image.open('static/' + img_fname))
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    batch = np.expand_dims(img, axis=0)
    batch_t_fsl = img_preprocess_fsl(batch)  # FSL Model
    batch_t_ssl = img_preprocess_ssl(batch)  # SSL Model
    # pred = model_fsl.predict(batch_t)    # FSL Model
    pred = tf.nn.softmax(model_ssl.predict(batch_t_ssl)).numpy()  # SSL Model
    if pred[0][1] > 0.5:
        diagnosis = 'PNEUMONIA'
        pred_prob = f'{pred[0][1] * 100:.0f}%'  # SSL Model
        # pred_prob = f'{pred[0][0] * 100:.0f}'  # FSL Model
    else:
        diagnosis = 'NORMAL'
        pred_prob = f'{pred[0][0] * 100:.0f}%'  # SSL Model
        # pred_prob = f'{(1 - pred[0][0]) * 100:.0f}'  # FSL Model
    prediction = f'P( {diagnosis} ) = {pred_prob}'
    print(f'{img_fname} ==> {pred_prob}')

    gen_heatmap_fsl(batch_t_fsl, img)
    gen_heatmap_ssl(batch_t_ssl, img)

    tsne.gen_tsne(batch_t_ssl, 'static/tsne.png')

    rand_refresh = str(int(time() % 8192))
    return templates.TemplateResponse("diagnose.html",
                                      {"request": request,
                                       "prediction": prediction,
                                       "f_example": f'{img_fname}?{rand_refresh}',
                                       "f_tsne": f'tsne.png?{rand_refresh}',
                                       "f_saliency1": f'saliency1.jpeg?{rand_refresh}',
                                       "f_saliency2": f'saliency2.jpeg?{rand_refresh}',
                                       })


def img_preprocess_fsl(imgs):
    imgs = tf.image.resize(imgs, IMG_SIZE_FSL[:2], method='bicubic')
    return imgs


def img_preprocess_ssl(imgs):
    imgs = tf.image.convert_image_dtype(imgs, dtype=tf.float32)
    imgs = tf.image.resize(imgs, IMG_SIZE_SSL[:2], method='bicubic')
    imgs = tf.clip_by_value(imgs, 0.0, 1.0)
    return imgs


def model_fsl_preprocess(x):
    x = Rescaling(1/255)(x)
    return x


def make_model_fsl(input_shape):
    base_model = ResNet50(include_top=False, pooling=None, weights=None,
                          input_shape=input_shape)

    inputs = Input(shape=input_shape)
    x = model_fsl_preprocess(inputs)
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
    # fully-supervised model
    global model_fsl
    model_fsl = make_model_fsl(IMG_SIZE_FSL)
    model_fsl.build(IMG_SIZE_FSL)
    model_fsl.load_weights('FSL_ResNet50_XrayReborn.h5')
    model_fsl.trainable = False

    # semi-supervised model
    global model_ssl
    model_ssl = tf.keras.models.load_model('model_ssl')
    model_ssl.build(IMG_SIZE_SSL)
    model_ssl.trainable = False

    global tsne
    # tsne = TSNE(model_fsl, 'dense', 'tsne_feats_fsl.npz')  # FSL
    tsne = TSNE(model_ssl, 'dense', 'tsne_feats_ssl.npz')  # SSL

    global gc_fsl, gc_ssl
    gc_fsl = GradCam(model_fsl, 'lambda_1', 'jet_colors.npy')  # FSL
    gc_ssl = GradCam(model_ssl, 'conv5_block3_out', 'jet_colors.npy')  # SSL


if __name__ == '__main__':
    main()
    uvicorn.run(app, host="0.0.0.0", port=8080)
else:
    main()
