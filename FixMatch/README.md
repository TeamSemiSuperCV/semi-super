# FixMatch Semi-Supervised Learning

[FixMatch](https://arxiv.org/abs/2001.07685) is an alternate method to perform semi-supervised learning with limited data labelling, also using Contrastive Learning.

We evaluated the performance of FixMatch versus SimCLR (with Fine-Tuning and Knowledge-Distillation) for 1%, 2% and 5% labels and found SimCLR to be faster and more accurate. Refer to the [Final Report](/docs/Final%20Report.pdf) for the comparison of results.

The code and instructions to run FixMatch on our dataset can be found in [FM.ipynb](FM.ipynb). It uses [fm.zip](fm.zip), which contains the FixMatch code (obtained from [Google Research](https://github.com/google-research/fixmatch/tree/dberth-tf2)) and data preparation and execution script, modified to work on our dataset.

