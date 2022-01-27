# Semi/Self-Supervised Learning on a Pediatric Pneumonia Dataset

## About
Fully supervised approaches need large, densely annotated datasets. Only hospitals that can afford to collect large annotated datasets can utilize these approaches to aid their physicians. The project goal is to utilize self-supervised and semi-supervised learning approaches to significantly reduce the need for fully labelled data. In this repo, you will find the project source code, along with training notebooks, and the final TensorFlow 2 saved model used to develop the web application for detecting Pediatric Pneumonia from chest X-rays.

The semi/self-supervised learning framework used in the project comprises of three stages: 

1. Self-supervised pretraining
2. Supervised fine-tuning
3. Knowledge distillation using unlabeled data

Refer to Google reserach team's paper (SimCLRv2 - Big Self-Supervised Models are Strong Semi-Supervised Learners) for more details regarding the framework used.

The example notebooks for the Stage 1, 2, 3 training can be found in the [notebooks](/notebooks) folder.

Resources

- [Chest X-ray Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2?__hstc=25856994.691713ea611804e2a755290a622023a7.1641825897692.1641825897692.1641825897692.1&__hssc=25856994.1.1641825897692&__hsfp=1000557398)
- [SimCLRv2 paper](https://arxiv.org/abs/2006.10029)

## ML workflow

<img src=".github\readme\Workflow.png" width=75% height=75%>

## App demo

<img src=".github\readme\SemiSuperCV.gif">

Follow the instructions below to run the app locally with Docker.

Once you have Docker installed, clone this repo 

```bash
git clone https://github.com/TeamSemiSuperCV/semi-super
```

Navigate to the webapp directory of the repo.

```bash
cd semi-super/webapp
```

Now build the container image using the `docker build` command. It will take few minutes.

```bash
docker build -t semi-super .
```

Start your container using the docker run command and specify the name of the image we just created:

```bash
docker run -dp 8080:8080 semi-super
```

After a few seconds, open your web browser to http://localhost:8080. You should see our app.

## Findings  

The project report and the final presentation slides can be found in the [docs](/docs) folder.
 
### Stage 1 - Contrastive Accuracy

| Labels | Stage 1 (self-supervised) |
| ------- | ------ | 
| No labels used | [99.99%](https://github.com/TeamSemiSuperCV/semi-super/blob/main/notebooks/Stage1/Stage1_self_supervised_training.ipynb) |

### Stage 2 and 3 - Test Accuracy Comparison

| Labels  | FSL[^1]    | Stage 2 (finetuning) | Stage 3 (distillation) | 
| ------- | ------ | -------------------- | ---------------------- | 
| 1% | [85.2%]() | [94.5%](https://github.com/TeamSemiSuperCV/semi-super/blob/main/notebooks/Stage2/Stage2_fine_tuning_1pct_labels.ipynb) | [96.3%](https://github.com/TeamSemiSuperCV/semi-super/blob/main/notebooks/Stage3/Stage3_distillation_1pct_labels.ipynb) | 
| 2% | [87.2%]() | [96.8%](https://github.com/TeamSemiSuperCV/semi-super/blob/main/notebooks/Stage2/Stage2_fine_tuning_2pct_labels.ipynb) | [97.6%](https://github.com/TeamSemiSuperCV/semi-super/blob/main/notebooks/Stage3/Stage3_distillation_2pct_and_5pct_labels.ipynb) | 
| 5% | [86.0%]() | [97.1%](https://github.com/TeamSemiSuperCV/semi-super/blob/main/notebooks/Stage2/Stage2_fine_tuning_5pct_labels.ipynb) | [98.1%](https://github.com/TeamSemiSuperCV/semi-super/blob/main/notebooks/Stage3/Stage3_distillation_2pct_and_5pct_labels.ipynb) | 
| 100% | [98.9%]() |  |  | 

[^1]: Fully-supervised model performance for benchmarking purposes.

## Acknowledgements

We have taken the SimCLR framework code from [Google Research](https://github.com/google-research/simclr) and have heavily modified it for the purpose of this project. We have added knowledge distillation feature along with several other changes. With these changes and improvements, now knowledge distillation can be performed on Google Cloud TPU cluster which reduces training time significantly. 
