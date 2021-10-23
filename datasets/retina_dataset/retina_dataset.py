"""retina_dataset dataset."""

import re
import os
import tensorflow_datasets as tfds

# TODO(retina_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION =  """\
Retinal OCT image dataset reflecting Drusen, DME, CNV and Normal 
"""

# TODO(retina_dataset): BibTeX citation
_CITATION = """\
title = {Retinal OCT image data}
author = {paultimothymooney}
publisher = {Kaggle}
url = {https://www.kaggle.com/paultimothymooney/kermany2018 }
"""

# _TRAIN_URL = "https://storage.googleapis.com/retinal_oct_archive/retinal_oct_train.zip"
# _TRAIN_URL = "https://storage.googleapis.com/retinal_oct_archive/retinal_oct_train_sm.zip"
# _TEST_URL = "https://storage.googleapis.com/retinal_oct_archive/retinal_oct_test.zip"

_TRAIN_FILE = "retinal_oct_train.zip"
_TEST_FILE = "retinal_oct_test.zip"

_LABELS = ["NORMAL", "DRUSEN", "DME", "CNV"]

_NAME_RE = re.compile(r"^(NORMAL|DRUSEN|DME|CNV)(?:/|\\)[\w-]*\.jpeg$")

class RetinaDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for retinaoct dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Download `retinal_oct_train.zip` and 'retinal_oct_test.zip' files from

  "https://storage.googleapis.com/retinal_oct_archive/retinal_oct_train_sm.zip"
  "https://storage.googleapis.com/retinal_oct_archive/retinal_oct_test.zip"

  Place the files in the `~/tensorflow_datasets/downloads/manual/` folder.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(),
            "label": tfds.features.ClassLabel(names=_LABELS)
        }),
        supervised_keys=("image", "label"),
        homepage='https://www.kaggle.com/paultimothymooney/kermany2018',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    # train_path, test_path = dl_manager.download(
    #     [_TRAIN_URL, _TEST_URL])

    train_path = os.path.join(dl_manager.manual_dir, _TRAIN_FILE)
    test_path = os.path.join(dl_manager.manual_dir, _TEST_FILE)

    return [
      tfds.core.SplitGenerator(
        name=tfds.Split.TRAIN,
        gen_kwargs={
          "archive": dl_manager.iter_archive(train_path)
        }),
      tfds.core.SplitGenerator(
        name=tfds.Split.TEST,
        gen_kwargs={
          "archive": dl_manager.iter_archive(test_path)
        }),
    ]

  def _generate_examples(self, archive):
    """Generate images and labels given the directory path.

    Args:
        archive: object that iterates over the zip.

    Yields:
        The image path and its corresponding label.
    """
    for fname, fobj in archive:
      # print('>', fname, fobj)
      res = _NAME_RE.match(fname)
      if not res:
        continue
      label = res.group(1)
      record = {
        "image": fobj,
        "label": label,
      }
      yield fname, record
