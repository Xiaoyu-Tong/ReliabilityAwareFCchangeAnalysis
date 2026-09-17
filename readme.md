# A Reliability-Aware Framework for Treatment-Induced Functional Connectivity (FC) Change Analysis

Code Repo for the accompanying paper [Early Brain Functional Connectivity Changes Induced by Antidepressants and Placebo](https://pmc.ncbi.nlm.nih.gov/articles/PMC12407754/), which demonstrates its applications in antidepressant treatment.
> Tong, Xiaoyu, Gregory A. Fonzo, Nancy B. Carlisle, Hua Xie, Yevgeny Berdichevsky, Corey J. Keller, Desmond J. Oathes, Charles B. Nemeroff, and Yu Zhang. "Early Brain Functional Connectivity Changes Induced by Antidepressants and Placebo." bioRxiv (2025).

## Workflow
<img src="/assets/Fig1.jpeg" width="1000">

The most straightforward way to characterize treatment-induced FC changes may be to simply calculate the pre-to-post-treatment change in FC features; however, this procedure may mix treatment effects with <ins>random fluctuations and measurement noise</ins>, especially when FC changes are assessed at the individual level. In this study, we address this challenge by introducing an explicit learning objective that <ins>maximizes the test-retest reliability</ins> of identified FC change patterns (**Fig. 1b**), as well as separating the overall FC changes into distinct interpretable dimensions (**Fig. 1a,c**).

## Formulation
Reliability-aware FC change analysis is a versatile framework that can be applied to either classification, regression, or unsupervised learning tasks. Essentially, it represents a family of learning objectives integrating task performance with reliability assessment. In this study, reliability of FC changes is encouraged by maximizing signal-to-noise ratio (SNR), where signal is defined as between-timepoint FC differences, and noise as within-timepoint test-retest FC differences. Namely, we sought FC change dimensions with significant, clinically meaningful between-timepoint differences and minimal test-retest variability.

### Supervised learning
In this study, reliability-aware predictive learning combines three learning objectives in the overall loss function -- SNR constraint, prediction task, and regularization:
```math
L = L_{SNR} + L_{task} + L_{reg}
```

Since the loss function is to be minimized, the SNR loss is defined as the ratio between noise power and signal power:
```math
L_{SNR} = {P_{noise}\over P_{signal}}
```

For classification, the task loss is:
```math
L_{task-classification} ={1\over N}\Bigl(\sum \ln{(e^{\Delta FC \cdot \beta} + 1)} - Y^\top \Delta FC \cdot \beta\Bigr)
```

For regression, the task loss is:
```math
L_{task-regression} = \lVert Y - \Delta FC \cdot \beta \rVert^2_2
```

L0-regularization is implemented on both FC dimension loadings and predictive weights to enhance generalizability and interpretability.

### Unsupervised learning
In this study, we used reliability-aware unsupervised learning to identify active-drug-specific FC changes -- this analysis involves no prediction tasks.
Essentially, we seek FC dimensions showing significant changes in the active drug arm and minimal changes in the placebo arm and test-retest differences. This learning objective can be formularized as:
```math
w = \arg\max_{\lVert w\rVert_2 = 1} {Var(\Delta FC_{drug})\over Var(\Delta FC_{placebo}) + Var(\Delta FC_{test-retest})}
```

> See corresponding Methods sections and Appendix A of the accompanying paper for math details.

## Project Structure

```text
ReliabilityAwareFCchangeAnalysis/
├── Codes/
│   ├── CVQ/  # Contrastive Variance Quotient Analysis (active-drug-specific FC change identification)
│   │   ├── CVQ.m  # Optimization algorithm for CVQ
│   │   └── demoMain_CVQ.m  # Conceptual demo of CVQ implementation
│   ├── Classification/
│   │   ├── optimizeClassification_ContinuousSparsification.m  # main optimization procedure
│   │   ├── optimizeClassification_FineTuning.m  # Auxiliary function for L0-regularization
│   │   ├── optimizeClassification_Initialization.m  # Initialization algorithm
│   │   └── updateG_Classification.m  # Auxiliary function for main optimization procedure
│   └── Regression/
│       ├── optimizeRegression_ContinuousSparsification.m  # main optimization procedure
│       ├── optimizeRegression_FineTuning.m  # Auxiliary function for L0-regularization
│       └── optimizeRegression_Initialization.m  # Initialization algorithm
├── assets/
│   └── Fig1.jpeg  # workflow figure
└── readme.md  # this file
```


## Optimization Pipelines

### Classification


### Regression


### Unsupervised learning (contrastive variance quotient analysis)


## Contact
tongxy@stanford.edu
