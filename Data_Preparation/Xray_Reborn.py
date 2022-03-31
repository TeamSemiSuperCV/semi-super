"""Xray_Reborn dataset."""

import tensorflow_datasets as tfds
from pathlib import Path
import re
import random

# TODO(Xray_Reborn): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(Xray_Reborn): BibTeX citation
_CITATION = """
@article{kermany2018identifying,
  title={Identifying medical diagnoses and treatable diseases by image-based deep learning},
  author={Kermany, Daniel S and Goldbaum, Michael and Cai, Wenjia and Valentim, Carolina CS and Liang, Huiying and Baxter, Sally L and McKeown, Alex and Yang, Ge and Wu, Xiaokang and Yan, Fangbing and others},
  journal={Cell},
  volume={172},
  number={5},
  pages={1122--1131},
  year={2018},
  publisher={Elsevier}
}
"""

DATA_DIR = '/content/XRay_'
CLASSES = ['NORMAL', 'PNEUMONIA']
PNEUMONIAS = ['VIRUS', 'BACTERIA']
CLASSES_EXT = CLASSES + PNEUMONIAS
FNAME_PAT = re.compile('^(.+?)-(.+)\.jpeg$')
RAND_SEED = 42

NUM_ORIG_TRAIN = 5232
NUM_ORIG_TEST = 623
NUM_ORIG_TOTAL = NUM_ORIG_TRAIN + NUM_ORIG_TEST

class XrayReborn(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Xray_Reborn dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(Xray_Reborn): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3), encoding_format='jpeg'),
            'label': tfds.features.ClassLabel(names=CLASSES),
            'fname': tfds.features.Text(),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(tfds): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('file:///Users/wing/ixig/FourthBrain/Midterm/Data.zip')
    path = Path(DATA_DIR)
    train_paths = list((path / 'train').glob('*/*.jpeg'))
    test_paths = list((path / 'test').glob('*/*.jpeg'))
    combined_paths = train_paths + test_paths
    random.seed(RAND_SEED)
    random.shuffle(combined_paths)
    train_paths = combined_paths[:len(train_paths)]
    test_paths = combined_paths[-len(test_paths):]
    assert len(train_paths) + len(test_paths) == len(combined_paths)
    # print('\n>>>', len(train_paths), len(test_paths), len(combined_paths))
    # raise Exception

    # TODO(tfds): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'test':       self._generate_examples(test_paths, 0.0, 1.0),
        'train':      self._generate_examples(train_paths, 0.0, 0.9),
        'validation': self._generate_examples(train_paths, 0.9, 1.0),
        'train_1pc':  self._generate_examples(train_paths, 0.0, 0.01),
        'train_2pc':  self._generate_examples(train_paths, 0.0, 0.02),
        'train_5pc':  self._generate_examples(train_paths, 0.0, 0.05),
        'train_10pc': self._generate_examples(train_paths, 0.0, 0.10),
    }

  def _generate_examples(self, paths, start, end):
    """Yields examples."""
    # TODO(tfds): Yields (key, example) tuples from the dataset
    total = len(paths)
    start_idx = int(start * total)
    end_idx = int(end * total)
    for f in paths[start_idx: end_idx]:
      matches = FNAME_PAT.match(f.name)
      klass = matches.group(1)
      assert klass in CLASSES_EXT
      klass = 'PNEUMONIA' if klass in PNEUMONIAS else 'NORMAL'
      key = '-'.join([f.parent.parent.name, f.parent.name, matches.group(1), matches.group(2)])
      # print(key)
      yield key, {
          'image': f,
          'label': klass,
          'fname': key,
      }
