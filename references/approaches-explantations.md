
# Learning Agency Lab - Automated Essay Scoring 2.0 Documentation

This project is my contribution to the Kaggle [Learning Agency Lab - Automated Essay Scoring 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2) competition.

## 1. Project Framing

There are several ways to approach this problem.

### 1.1 Ordinal Classification

Ordinal classification is an ideal approach since our target values are ordinal integers. However, this presents a challenge during training when defining the appropriate loss function and metric to optimize.

This is the default approach proposed by the competition, where the **Quadratic Weighted Kappa** is used as the evaluation metric, and **binary cross-entropy** is suggested as the loss function.

### 1.2 Multiclass Classification

Alternatively, the grades can be framed as a multi-class classification problem. In this case, the training and evaluation phases become more straightforward because we already have well-established metrics and loss functions for this approach.

### 1.3 Regression

Choosing to frame this as a regression problem implies that our model will output real-valued predictions on the [1-6] scale. However, since the scores are supposed to be integers, we would need to either round the predictions or find optimal threshold values between intervals (e.g., if $\hat{y}$ > 1.4, then $\hat{y} = 2$, else $\hat{y} = 1$).

### Chosen Approach

Among these three approaches, we selected **ordinal classification** because it aligns with the competition's evaluation metric, which determines how our submission is scored and ranked.

## 2. Training Approaches

Our goal is to build a model that can accurately score student essays on the 1 to 6 scale. Leveraging recent advances in Natural Language Processing (NLP), we explored various methods to solve this problem.

### 2.1 Training Approach 1: Fine-Tune a Large Language Model (LLM)

LLMs are pre-trained on vast datasets. Through techniques like **Masked Language Modeling**, LLMs develop bidirectional contextual representations of words, essentially gaining a statistical understanding of language.

During the fine-tuning process, we leverage this pre-learned language understanding to make the LLM perform well on our specific task (e.g., classification, sentiment analysis).

In this project, we experimented with fine-tuning different variants of the [DeBERTa model](https://keras.io/api/keras_nlp/models/deberta_v3/#debertav3backbone-model), adjusting hyperparameters such as `batch_size` and `max_sequence_length`, as suggested in the competition's starting notebook.

In essence, we used a pre-trained DeBERTa backbone model and added a Dense layer for classification. This was easily implemented using the [DeBERTaV3Classifier](https://keras.io/api/keras_nlp/models/deberta_v3/deberta_v3_text_classifier/).

### 2.2 Use Pre-Trained Embedding Models

In this approach, we utilized pre-trained embedding models from Hugging Face to generate high-dimensional vectors representing the meaning of the essays. These embedding vectors were then used as input to a Deep Neural Network with a classifier at the final layer.

Since many embedding models are available, we referenced the [Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) to select the best model for our task (classification, language: English, with a model size limited to hundreds of millions of parameters due to GPU VRAM constraints).

--------
