# IMDB Movie Review Sentiment Analysis with an RNN

Binary sentiment classification on 50,000 IMDB movie reviews. Text is cleaned and vectorised with TF-IDF, then classified by a recurrent neural network built in PyTorch.

**Test accuracy: 85.7%**

## Dataset

The IMDB Dataset of 50K Movie Reviews, with one review text column and a positive or negative sentiment label. The file is not stored in this repository because of its size. Download it from Kaggle and save it as `IMDB Dataset.csv` in the project root, next to the notebook.

After removing 418 duplicate rows, 49,582 reviews remain.

## Pipeline

1. Remove duplicates and check for missing values
2. Lowercase the text and strip URLs, punctuation and HTML tags
3. Remove English stopwords using NLTK
4. Apply Porter stemming
5. Encode labels and build TF-IDF features with a vocabulary of 5,000 terms
6. Split the data 80/20 into 39,665 training and 9,917 test reviews
7. Train a single-layer `nn.RNN` with 128 hidden units, followed by a linear output layer
8. Optimise with Adam and binary cross-entropy for 10 epochs, batch size 64
9. Evaluate accuracy on the held-out test set

## Results

| Metric | Value |
|---|---|
| Test accuracy | 85.70% |
| Training samples | 39,665 |
| Test samples | 9,917 |
| Epochs | 10 |

## Installation

```
pip install -r requirements.txt
jupyter lab
```

Open `RNN_for_sentimentanalysis.ipynb` and run the cells in order. The required NLTK resources (`punkt_tab`, `stopwords`) are downloaded inside the notebook.

## Future Improvements

The current model treats each TF-IDF vector as a single time step, so it does not yet make use of word order. The next step is to tokenise the reviews, use learned word embeddings and feed the actual word sequence to an LSTM or GRU. Stopword removal can also be made token-based so that words are never altered during cleaning, and precision, recall and a confusion matrix can be added to the evaluation.
