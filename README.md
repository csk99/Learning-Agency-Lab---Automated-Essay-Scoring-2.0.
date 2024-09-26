# Learning Agency Lab - Automated Essay Scoring 2.0


This project is my contribution to a kaggle Learning Agency Lab - Automated Essay Scoring 2.0 <a href="https://www.kaggle.com/competitions learning-agency-lab-automated-essay-scoring-2">competition</a>.



<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>


The goal of this competition is to train a model to score student essays on a 1 to 6 scale. The competition's evaluation metric is the **quadratic weighted kappa**, which measures the agreement between two outcomes.

The quadratic weighted kappa is calculated as follows. First, an \( N \times N \) histogram matrix \( O \) is constructed, where \( O_{i,j} \) corresponds to the number of essay IDs with an actual score \( i \) that received a predicted score \( j \).

Next, an \( N \times N \) matrix of weights \( w \) is calculated based on the difference between the actual and predicted values:

\[
w_{i,j} = \frac{(i - j)^2}{(N - 1)^2}
\]

An \( N \times N \) histogram matrix of expected outcomes, \( E \), is then calculated, assuming no correlation between the actual and predicted values. This is done by computing the outer product of the actual outcomes vector and the predicted outcomes vector, normalized such that \( E \) and \( O \) have the same sum.

From these three matrices, the quadratic weighted kappa is calculated as:

\[
\kappa = 1 - \frac{\sum_{i,j} w_{i,j} O_{i,j}}{\sum_{i,j} w_{i,j} E_{i,j}}
\]



## Technologies / Frameworks used 
* ![Static Badge](https://img.shields.io/badge/Python-3.10-green)
* ![Static Badge](https://img.shields.io/badge/keras_nlp-0.9.3-green)
* ![Static Badge](https://img.shields.io/badge/keras-3.2.1-green)


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── train.csv       <- Training data 
│   ├── test.csv        <- Test data.
│   └──sample_submission.csv      <- The final prediction to be reviewed / scored.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         learning_agency_lab___automated_essay_scoring_2.0 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── learning_agency_lab___automated_essay_scoring_2.0   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes learning_agency_lab___automated_essay_scoring_2.0 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

