# Protection of AI Models Against Adversarial Attacks

This project presents an experimental analysis of adversarial attacks and defense mechanisms on object detection models, with a focus on the YOLOv8 architecture.

## Overview

The goal of this project is to evaluate the vulnerability of deep learning models to adversarial perturbations and to assess the effectiveness of adversarial training as a defense strategy.

The experiments are conducted on a human detection task using the HERIDAL dataset.

## Key Features

- Implementation of **FGSM (Fast Gradient Sign Method)** adversarial attack
- Evaluation under multiple perturbation strengths (ε values)
- Analysis of model performance using:
  - mAP50 and mAP50-95
  - Precision and Recall
  - TP, FP, FN metrics
- **Adversarial training** as a defense mechanism
- Additional robustness evaluation using **PGD (Projected Gradient Descent)** attack

## Results

- Adversarial attacks significantly degrade detection performance, especially Recall
- The model tends to **miss real objects** rather than generate false positives
- Adversarial training improves robustness, but does not fully eliminate vulnerability
- The trained model shows partial robustness even against stronger PGD attacks

## Dataset

- HERIDAL (Human detection in difficult environments)

## Usage

1. Train the baseline model  
2. Generate adversarial examples using FGSM  
3. Evaluate model performance under different ε values  
4. Apply adversarial training  
5. Test robustness with FGSM and PGD attacks  

## Notes

This project is part of a research study on adversarial robustness in computer vision systems.

## Authors

- Marijana Bandić  
- Maja Kovačić  
- Fran Pavlović
