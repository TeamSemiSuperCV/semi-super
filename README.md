# Semi / Self-Supervised Learning on a Pediatric Pneumonia Dataset

## About
Fully supervised approaches need large, densely annotated datasets. Only hospitals that can afford to collect large annotated datasets can utilize these approaches to aid their physicians. The project goal is to utilize self-supervised and semi-supervised learning approaches to significantly reduce the need for fully labelled data. In this repo, you will find the project source code, along with training notebooks, and the final TensorFlow 2 saved model used to develop the web application for detecting Pediatric Pneumonia from chest X-rays.

The semi/self-supervised learning framework used in the project comprises of three stages: 

1. Self-supervised pretraining
2. Supervised fine-tuning with active-learning
3. Knowledge distillation using unlabeled data

Refer to Google reserach team's paper (SimCLRv2 - Big Self-Supervised Models are Strong Semi-Supervised Learners) for more details regarding the framework used.

The training notebooks for Stage 1, 2, and 3 can be found in the [notebooks](/notebooks) folder. Notebooks for Selective Labeling (active-learning) using Entropy or Augmentations policies can be found in the [Active_Learn](/Active_Learn) folder. Benchmarks for Fully-Supervised Learning can be found in the [FSL_Benchmarks](/FSL_Benchmarks) folder. The code for Data Preprocessing can be found in [Data_Preparation](/Data_Preparation).

- [Project Report](/docs/Final%20Report.pdf)
- [Final Presentation](/docs/FinalPresentation.pdf)
- [FourthBrain Award](/docs/FourthBrainTopProjectAward.pdf)

## Results  
 
### Stage 1 - Contrastive Accuracy

| Labels | Stage 1 (Self-Supervised) |
| ------- | ------ | 
| No labels used | [99.99%](/notebooks/Stage1/Stage1_self_supervised_training.ipynb) |

### Stage 2 and 3 - Test Accuracy Comparison

| Labels | FSL (Benchmark) | Stage 2 (Finetuning) | Stage 3 (Distillation) | 
| ------- | ------ | -------------------- | ---------------------- | 
| 1% | [85.2%](/FSL_Benchmarks/FSL_ResNet50_XrayReborn_1pc.ipynb) | [94.5%](/notebooks/Stage2/Stage2_fine_tuning_1pct_labels.ipynb) | [96.3%](/notebooks/Stage3/Stage3_distillation_1pct_labels.ipynb) | 
| 2% | 87.2% | [96.8%](/notebooks/Stage2/Stage2_fine_tuning_2pct_labels.ipynb) | [97.6%](/notebooks/Stage3/Stage3_distillation_2pct_and_5pct_labels.ipynb) | 
| 5% | [86.0%](/FSL_Benchmarks/FSL_ResNet50_XrayReborn_5pc.ipynb) | [97.1%](/notebooks/Stage2/Stage2_fine_tuning_5pct_labels.ipynb) | [98.1%](/notebooks/Stage3/Stage3_distillation_2pct_and_5pct_labels.ipynb) | 
| 100% | [98.9%](/FSL_Benchmarks/FSL_ResNet50_XrayReborn.ipynb) | N/A | N/A | 

Despite needing only a small fraction of labels, our Stage 2 and Stage 3 models were able to acheive test accuracies that are comparable to a 100% labelled Fully-Supervised (FSL) model. Refer to the [Project Report](/docs/Final%20Report.pdf) and the [Final Presentation](/docs/FinalPresentation.pdf) for a more detailed discussion and findings.

## ML Workflow

<img src=".github\readme\Workflow.png" width=75% height=75%>

## Web App Demo

<img src=".github\readme\SemiSuperCV.gif">

## Installation

Your can run the app locally if you have Docker installed. First, clone this repo:
```bash
git clone https://github.com/TeamSemiSuperCV/semi-super
```

Navigate to the webapp directory of the repo:
```bash
cd semi-super/webapp
```

Build the container image using the `docker build` command (will take few minutes):
```bash
docker build -t semi-super .
```

Start the container using the `docker run` command, specifying the name of the image we just created:
```bash
docker run -dp 8080:8080 semi-super
```

After a few seconds, open your web browser to http://localhost:8080. You should see the app.

## Acknowledgements

We took the SimCLR framework code from [Google Research](https://github.com/google-research/simclr) and heavily modified it for the purposes of this project. We enhanced the knowledge distillation feature along with several other changes to make it perform better with our dataset. With these changes and improvements, knowledge distillation can be performed on the Google Cloud TPU infrastructure, which reduces training time significantly. 

## Other Resources

- [Chest X-ray Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2?__hstc=25856994.691713ea611804e2a755290a622023a7.1641825897692.1641825897692.1641825897692.1&__hssc=25856994.1.1641825897692&__hsfp=1000557398)
- [SimCLRv2 Paper](https://arxiv.org/abs/2006.10029)
