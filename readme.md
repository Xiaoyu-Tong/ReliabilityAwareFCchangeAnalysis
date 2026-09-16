# A Reliability-Aware Framework for Treatment-Induced Functional Connectivity (FC) Change Analysis

Code Repo for the accompanying paper [Early Brain Functional Connectivity Changes Induced by Antidepressants and Placebo](https://pmc.ncbi.nlm.nih.gov/articles/PMC12407754/), which demonstrates its applications in antidepressant treatment.
> Tong, Xiaoyu, Gregory A. Fonzo, Nancy B. Carlisle, Hua Xie, Yevgeny Berdichevsky, Corey J. Keller, Desmond J. Oathes, Charles B. Nemeroff, and Yu Zhang. "Early Brain Functional Connectivity Changes Induced by Antidepressants and Placebo." bioRxiv (2025).

## Workflow
<img src="/assets/Fig1.jpeg" width="1000">

The most straightforward way to characterize treatment-induced FC changes may be to simply calculate the pre-to-post-treatment change in FC features; however, this procedure may mix treatment effects with <ins>random fluctuations and measurement noise</ins>, especially when FC changes are assessed at the individual level. In this study, we address this challenge by introducing an explicit learning objective that <ins>maximizes the test-retest reliability</ins> of identified FC change patterns (**Fig. 1b**), as well as separating the overall FC changes into distinct interpretable dimensions (**Fig. 1a,c**).

### Formulation

