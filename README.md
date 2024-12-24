# Automated-Issue-Classification
The exponential growth of open-source software
projects has led to an increasing volume of issues reported
on platforms like GitHub. These issues span diverse categories,
such as bug reports, feature requests, and general inquiries,
posing challenges for efficient triaging and classification. Manual
classification is labor-intensive and error-prone, especially for
large-scale repositories. This study explores automated issue
classification using machine learning (ML) and deep learning
(DL) techniques, leveraging contextual language models such
as RoBERTa and Sentence Transformers for feature extraction.
Key contributions include the development of a framework
integrating advanced embeddings with classifiers like Support
Vector Machines (SVM), Random Forests (RF), LightGBM,
XGBoost, Long Short-Term Memory networks (LSTM), and
DistilBERT. Empirical evaluations on a balanced dataset of
3,000 GitHub issues demonstrate that DistilBERT, paired with
RoBERTa and Sentence Transformer embeddings, achieves near-
perfect classification performance (F1-score: 0.992). This work
highlights the trade-offs between computational efficiency and
accuracy, and underscore the transformative potential of pre-
trained transformer models for issue classification. The study
establishes a foundation for future research, advocating for
enhanced dataset diversity and the exploration of lightweight,
domain-specific transformer models.
