# Fully-Supervised Learning Benchmark Results

| FileNames | Description |
| --- | --- |
| FSL_ResNet50_XrayOrig | Original Dataset Distribution |
| FSL_ResNet50_XrayReborn | Equalized Dataset Distribution * |

*: The authors of the dataset intentionally chose distinct and different sampling distributions between their 'Test' vs 'Train' images. Since machine learning models are constructed under the assumption that the training data is governed by the exact same distribution over which the model will later be evaluated upon, and that the goal of the project is not to gauge how well our models can handle data drift, but rather to *compare the performance of Fully-Supervised vs Semi-Supervised Learning approaches*, we combined the original dataset's (*XrayOrig*) Test and Train subsets and then resampled both subsets to come from this equalized distribution (*XrayReborn*).
