# A Reliability-Aware Framework for Treatment-Induced Functional Connectivity (FC) Change Analysis

Code Repo for the accompanying paper [Early Brain Functional Connectivity Changes Induced by Antidepressants and Placebo](https://pmc.ncbi.nlm.nih.gov/articles/PMC12407754/), which demonstrates its applications in antidepressant treatment.
> Tong, Xiaoyu, Gregory A. Fonzo, Nancy B. Carlisle, Hua Xie, Yevgeny Berdichevsky, Corey J. Keller, Desmond J. Oathes, Charles B. Nemeroff, and Yu Zhang. "Early Brain Functional Connectivity Changes Induced by Antidepressants and Placebo." bioRxiv (2025).

## Workflow
<img src="/assets/Fig1.jpeg" width="1000">

The most straightforward way to characterize treatment-induced FC changes may be to simply calculate the pre-to-post-treatment change in FC features; however, this procedure may mix treatment effects with <ins>random fluctuations and measurement noise</ins>, especially when FC changes are assessed at the individual level. In this study, we address this challenge by introducing an explicit learning objective that <ins>maximizes the test-retest reliability</ins> of identified FC change patterns (**Fig. 1b**), as well as separating the overall FC changes into distinct interpretable dimensions (**Fig. 1a,c**).

## Formulation
Reliability-aware FC change analysis is a versatile framework that can be applied to either classification, regression, or unsupervised learning tasks. Essentially, it represents a family of learning objectives integrating task performance with reliability assessment. In this study, reliability of FC changes is encouraged by maximizing signal-to-noise ratio (SNR), where signal is defined as between-timepoint FC differences, and noise as within-timepoint test-retest FC differences. Namely, we sought FC change dimensions with significant, clinically meaningful between-timepoint differences and minimal test-retest variability.

### The classification case


### The regression case


### The unsupervised learning case



> See Appendix A of the accompanying paper for math details.

## Project Structure

## Pipelines

### Classification


### Regression


### Unsupervised learning (contrastive variance quotient analysis)


## Contact
tongxy@stanford.edu
