import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from openTSNE import TSNE as OTS_TSNE
from tensorflow.keras.models import Model

PERPLEXITY = 12
RAND_STATE = 42


class TSNE():

    def __init__(self, model, layer_name):
        feat_output = model.get_layer(layer_name).output
        self.model = Model(inputs=model.input, outputs=feat_output)

        # load existing features
        tsne_npz = np.load('tsne_feats.npz')
        self.tsne_features = tsne_npz['feats']
        self.tsne_labels = tsne_npz['labels']
        # add an extra label for this image (we want a it a different color)
        self.new_label = np.array(
            [np.max(self.tsne_labels) + 1], dtype=np.int32)
        # pre-fit t-SNE to existing features
        tsne = OTS_TSNE(perplexity=PERPLEXITY, n_jobs=4,
                        metric="euclidean", random_state=RAND_STATE)
        self.embeddings = tsne.fit(self.tsne_features)

    def gen_tsne(self, batch_t, fname):
        '''
        generates 2d static t-SNE image as a PNG file
        '''
        labels, embeddings = self.get_tsne(batch_t)

        # PLOT THE RESULTS
        sns.set(font_scale=1.5)
        sns.set_style("white")
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
        log_0 = labels == 0
        log_1 = labels == 1
        log_2 = labels == 2
        plt.scatter(x=embeddings[log_0, 0], y=embeddings[log_0, 1],
                    c='#30a2da',  # c='#e5ae38', #c= '#fc4f30', #c='gray', #c='tomato',
                    s=20, alpha=0.5, label='Normal')
        plt.scatter(x=embeddings[log_1, 0], y=embeddings[log_1, 1],
                    c='#e5ae38',  # c='#fc4f30', #c='#6d904f', #c= '#30a2da', #c='red', #c='royalblue',
                    s=20, alpha=0.5, label='Pneumonia')
        plt.scatter(x=embeddings[log_2, 0], y=embeddings[log_2, 1],
                    c='black',
                    s=200, alpha=1, marker='*', label='Diagnosed')
        # sns.scatterplot(x=embeddings[:,0], y=embeddings[:,1], hue=labels, ax=ax,size=dotsize, alpha=.8, palette="colorblind")
        sns.despine()
        lim = (embeddings.min()-5, embeddings.max()+5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.set_title('t-SNE')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0.0, frameon=False);
        ax.legend(loc='best', borderaxespad=0.0, frameon=False,
                  labelspacing=0.25, handletextpad=-.25)
        plt.savefig(fname)

    def get_tsne(self, batch_t):
        ''' 
        gets the features from the dense layers and run t-SNE transform
        returns (labels, embeddings)
        '''
        # get the output of the dense layer for this particular image
        new_feat = self.model.predict(batch_t)
        # t-SNE transform features to get new embedding
        new_embedding = self.embeddings.transform(new_feat)
        # tack new embedding and label to existing
        final_labels = np.concatenate(
            (self.tsne_labels, self.new_label), axis=0)
        final_embeddings = np.concatenate(
            (self.embeddings, new_embedding), axis=0)
        return (final_labels, final_embeddings)
