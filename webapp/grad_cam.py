import numpy as np
import tensorflow as tf


class GradCam():

    def __init__(self, model, last_conv_layer,
                 cmap_fpath, alpha=0.35, beta=0.7) -> None:
        self.model = model
        if isinstance(last_conv_layer, str):
            self.last_conv_layer = self.model.get_layer(last_conv_layer)
        else:
            self.last_conv_layer = last_conv_layer
        self.alpha = tf.constant(alpha)
        self.beta = tf.constant(beta)
        self.jet_colors = np.load(cmap_fpath)

    def make_heatmap(self, img_batch, pred_index):
        # Create Grad-CAM heatmap of single image
        assert len(img_batch.shape) == 4 and img_batch.shape[0] == 1

        # Grad-CAM requires model's output to be logits
        last_activation = self.model.layers[-1].activation
        self.model.layers[-1].activation = None

        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        self.grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.last_conv_layer.output, self.model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.grad_model(img_batch)
            if pred_index is None:  # then pick the highest-probability class
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # Restore model's output activation
        self.model.layers[-1].activation = last_activation

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap

    def gen_grayscale_heatmap(self, img_batch, img, pred_index):
        assert img_batch.dtype == tf.float32 and img_batch.shape[-1] == 3
        assert img.dtype == np.uint8 and \
            len(img.shape) == 3 and img.shape[-1] == 3

        heatmap = self.make_heatmap(img_batch, pred_index)

        heatmap_rgb = tf.stack([heatmap] * 3, axis=-1)

        heatmap_rsg = tf.image.resize(
            heatmap_rgb, img.shape[:2], tf.image.ResizeMethod.MITCHELLCUBIC)
        heatmap_rsg = tf.clip_by_value(heatmap_rsg, 0.0, 1.0)
        heatmap_rsg = heatmap_rsg * 1.15 + 0.05

        superimp_g = heatmap_rsg * img
        superimp_g = tf.cast(
            tf.clip_by_value(superimp_g, 0.0, 255.0),
            tf.uint8)
        return superimp_g

    def gen_color_heatmap(self, img_batch, img, pred_index):
        assert img_batch.dtype == tf.float32 and img_batch.shape[-1] == 3
        assert img.dtype == np.uint8 and \
            len(img.shape) == 3 and img.shape[-1] == 3

        heatmap = self.make_heatmap(img_batch, pred_index)

        heatmap_ui8 = np.uint8(255 * heatmap)
        heatmap_jet = self.jet_colors[heatmap_ui8]

        heatmap_rsc = tf.image.resize(
            heatmap_jet, img.shape[:2], tf.image.ResizeMethod.MITCHELLCUBIC)
        heatmap_rsc = tf.clip_by_value(heatmap_rsc, 0.0, 1.0)

        superimp_c = heatmap_rsc * self.alpha + img/255 * self.beta
        superimp_c = tf.cast(
            tf.clip_by_value(superimp_c * 255, 0.0, 255.0),
            tf.uint8)
        return superimp_c
