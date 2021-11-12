import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as SKL_TSNE
from tensorflow.keras.models import Model


PERPLEXITY = 25


class TSNE():

    def __init__(self, model, layer_name):
        feat_output = model.get_layer(layer_name).output
        self.model = Model(inputs=model.input, outputs=feat_output)

        # load existing features
        tsne_npz = np.load('tsne_feats.npz')
        self.tsne_features = tsne_npz['feats']
        self.tsne_labels = tsne_npz['labels']
        # add an extra label for this image (we want a it a different color)
        self.this_label = np.max(self.tsne_labels) + 1
        self.pca = PCA(n_components=25).fit(self.tsne_features)
        self.tsne = SKL_TSNE(perplexity=PERPLEXITY, n_components=2,
                             metric='euclidean', random_state=2).fit(self.tsne_features)

    def gen_tsne(self, batch_t, fname):
        '''
        generates 2d static tse image as a PNG file 
        '''
        labels, features = self.get_feats(batch_t)
        # RUN PCA
        x = self.pca.transform(features)
        # RUN TSNE
        tsne_result = self.tsne.transform(x)

        # PLOT THE RESULTS
        sns.set(font_scale=1.5)
        sns.set_style("white")
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
        log_0 = labels == 0
        log_1 = labels == 1
        log_2 = labels == 2
        plt.scatter(x=tsne_result[log_0, 0], y=tsne_result[log_0, 1],
                    c='#30a2da',  # c='#e5ae38', #c= '#fc4f30', #c='gray', #c='tomato',
                    s=20, alpha=0.5, label='Normal')
        plt.scatter(x=tsne_result[log_1, 0], y=tsne_result[log_1, 1],
                    c='#e5ae38',  # '#fc4f30', #c='#6d904f', #c= '#30a2da', #c='red', #c='royalblue',
                    s=20, alpha=0.5, label='Pneumonia')
        plt.scatter(x=tsne_result[log_2, 0], y=tsne_result[log_2, 1],
                    c='black',
                    s=200, alpha=1, marker='*', label='Diagnosed')
        # sns.scatterplot(x=tsne_result[:,0], y=tsne_result[:,1], hue=labels, ax=ax,size=dotsize,alpha=.8, palette="colorblind")
        sns.despine()
        lim = (tsne_result.min()-5, tsne_result.max()+5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.set_title('tSNE')
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0.0, frameon=False);
        ax.legend(loc='best', borderaxespad=0.0, frameon=False,
                  labelspacing=0.25, handletextpad=-.25)
        plt.savefig(fname)

    def get_feats(self, batch_t):
        ''' 
        gets the features from the dense layers,
        returns (labels, features, thisfeat)
        '''
        # get the output of the dense layer for this particular image
        thisfeat = self.model.predict(batch_t)
        # add this image's data to (labels, features)
        labels = np.append(self.tsne_labels, self.this_label)
        features = np.concatenate((self.tsne_features, thisfeat), axis=0)
        return (labels, features)
