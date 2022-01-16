# Semi/Self-Supervised Learning on a Pediatric Pneumonia Dataset

## About
Fully supervised approaches need large, densely annotated datasets. Only hospitals that can afford to collect large annotated datasets can utilize these approaches to aid their physicians. The project goal is to utilize Semi/Self-Supervised Learning to significantly reduce the need for fully labelled data. In this repo you will find the source code, along with training notebooks, and the final TensorFlow 2 saved model used to create the web application for detecting Pediatric Pneumonia from chest X-rays.

The Semi-supervised learning framework used in the project comprises of three steps as shown in the workflow diagram below: 

Markup : 1. Unsupervised or self-supervised pretraining
         2. Supervised fine-tuning
         3. Knowledge distillation using unlabeled data

Resources

- [Chest X-ray Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2?__hstc=25856994.691713ea611804e2a755290a622023a7.1641825897692.1641825897692.1641825897692.1&__hssc=25856994.1.1641825897692&__hsfp=1000557398)

## ML workflow

<img src="Workflow.png" width=75% height=75%>

## Acknowledgements

- [SimCLRv2](https://arxiv.org/abs/2006.10029)
