# Conformal Prediction and the Many-Worlds of Fairness

Master’s Thesis by Julia Sophie Broden <br>  
LMU Munich, Department of Statistics <br>
Supervised by Prof. Dr. Christoph Kern <br>
Submitted: August 23, 2025


## Abstract

Fairness in Algorithmic Decision-Making is a matter of growing societal importance. However, standard fairness assessments often overlook the influence of design decisions. These choices can lead to a multiplicity of plausible models, each with different fairness outcomes. In addition, most existing methods omit the role of uncertainty, which can systematically vary across demographic groups. 

This thesis introduces a novel framework, Conformal Multiverse Analysis (CMA), which integrates Conformal Prediction with Multiverse Analysis. CMA enables a comprehensive assessment of how uncertainty propagates across different modeling choices and affects group-conditional coverage of different groups and minorities. The framework provides systematic procedures for identifying and potentially mitigating disparities from model design decisions.

Uncertainty is quantified through the Average Prediction Set Size (APSS). At the same time, fairness is assessed via Group-Conditional Coverage (GCC), focusing on non-German female jobseekers as a minority group. The resulting outcome distributions are analyzed using functional ANOVA and Lasso regression, capturing both main effects and interactions.
The results show that ensemble models such as random forests and gradient boosting machines produce smaller prediction sets (lower APSS), indicating higher confidence, but at the same time tend to under-cover minority groups, reducing fairness in GCC. Conversely, linear models like logistic regression and elastic net often yield wider sets and higher coverage for minority groups, suggesting a trade-off between confidence and fairness. Importantly, design decisions interact in complex ways. These findings underscore that fairness and uncertainty are not aligned by default and are shaped by subtle interactions between methodological choices.

## Overview

This repository builds on the code from the paper [One Model Many Scores: Using Multiverse Analysis to Prevent Fairness Hacking and Evaluate the Influence of Model Design Decisions](https://dl.acm.org/doi/10.1145/3630106.3658974) by Jan Simson, Florian Pfisterer and Christoph Kern, published in the proceedings of the ACM Conference on Fairness, Accountability, and Transparency 2024 in Rio de Janeiro, Brazil in June 2024. We adapted their framework to our setting of Conformal Multiverse Analysis (CMA).

## Setup

This project uses [Pipenv](https://pipenv.pypa.io/en/latest/) to control the Python environment. To install the dependencies, first install pipenv on your machine, then run `pipenv sync -d` in the root directory of the project. Once set up, you can enter the virtual environment in your command line by running `pipenv shell`.

## Files and Directories

- The [universe_analysis.ipynb](https://github.com/jsbroden/CMA_Fairness/blob/main/universe_analysis.ipynb) notebook contains all options examined in the multiverse. It runs a single universe/configuration.
- The [multiverse_analysis.py](https://github.com/jsbroden/CMA_Fairness/blob/main/multiverse_analysis.py) script includes all available options for the decisions in the universe_analysis.ipynb. Execute the multiverse analysis script by running `python multiverse_analysis.py`. Results will be saved in the [output](https://github.com/jsbroden/CMA_Fairness/tree/main/output) folder. 
- The folder [fairness_multiverse](https://github.com/jsbroden/CMA_Fairness/tree/main/fairness_multiverse) contains objects, classes, and helper functions for the CMA.  
- The [analysis_var_imp_overall.ipynb](https://github.com/jsbroden/CMA_Fairness/blob/main/analysis_var_imp_overall.ipynb) contains the Lasso Regression and FANOVA code. 
