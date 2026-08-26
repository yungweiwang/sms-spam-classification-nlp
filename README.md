# SMS Spam Classification with Classical NLP

## Overview
This project aims to classify SMS messages as spam or ham using classical natural language processing (NLP) and machine learning techniques. It compares Bag of Words (BoW), TF-IDF, and n-gram text representations using Logistic Regression as the classifier. In addition to model comparison, the project uses error analysis to identify weaknesses in text preprocessing and applies feature engineering to improve spam detection.

## Dataset
This project uses the SMS Spam Collection Dataset, which contains 5,572 SMS messages labeled as either ham or spam. The dataset consists of 4,825 ham messages and 747 spam messages. This substantial class imbalance is important when evaluating model performance because accuracy alone may provide a misleading picture of the model's ability to detect spam.  

## Methods
The text preprocessing pipeline includes tokenization, lowercasing, punctuation removal, stopword removal, and stemming. After preprocessing, the text is transformed into numerical features using Bag of Words (BoW), TF-IDF, and n-gram representations. Logistic Regression is used as the classifier across the experiments to provide a consistent comparison between different text representations. Model performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

## Experiments
Five experimental settings were compared:

1. **BoW Unigram:** Represents each message using individual word-frequency features.
2. **BoW + Bigrams:** Extends the BoW representation by including both individual words and adjacent two-word sequences.
3. **TF-IDF Unigram:** Represents individual words using TF-IDF weights, which reflect both their frequency within a message and their rarity across the corpus.
4. **TF-IDF + Bigrams:** Applies TF-IDF weighting to both individual words and adjacent two-word sequences.
5. **BoW + NUMBER:** Extends the unigram BoW model by normalizing digit sequences to a `NUMBER` token in order to preserve numeric information that was removed during the original preprocessing pipeline.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| BoW Unigram | 0.9812 | 0.9776 | 0.8792 | 0.9258 |
| BoW + Bigrams | 0.9767 | 0.9920 | 0.8322 | 0.9051 |
| TF-IDF Unigram | 0.9614 | 1.0000 | 0.7114 | 0.8314 |
| TF-IDF + Bigrams | 0.9587 | 0.9905 | 0.6980 | 0.8189 |
| **BoW + NUMBER** | **0.9839** | 0.9712 | **0.9060** | **0.9375** |

The BoW + NUMBER model achieved the best overall performance, with the highest accuracy (0.9839), recall (0.9060), and F1-score (0.9375) among the evaluated models.

Although the TF-IDF unigram model achieved the highest precision (1.0000), its substantially lower recall (0.7114) indicates that it failed to detect many spam messages.

## Error Analysis and Feature Engineering
Error analysis revealed that several spam messages were misclassified as ham because potentially informative numeric information was removed during the original preprocessing pipeline. Further analysis showed that 94.78% of spam messages contained digits, compared with only 15.65% of ham messages.

To preserve this information, digit sequences were normalized to a `NUMBER` token using regular expressions. After this feature-engineering step, spam recall increased from 0.8792 to 0.9060, while the number of false negatives decreased from 18 to 14.

## Key Findings
- **TF-IDF is not necessarily superior to Bag of Words for text classification.** In this experiment, BoW achieved better overall performance, showing that a more sophisticated weighting scheme does not necessarily lead to better classification results.

- **Preprocessing decisions can substantially affect model performance.** Removing non-alphabetic information initially discarded numeric cues that were strongly associated with spam messages.

- **Error analysis can guide meaningful model improvements.** Examining misclassified messages revealed weaknesses in the preprocessing pipeline and motivated numeric feature engineering, which improved spam recall and reduced false negatives.

## Limitations
The BoW + Logistic Regression model does not explicitly capture word order, broader context, or semantic information. Instead, it relies primarily on surface-level lexical features, which may cause messages with unusual wording or patterns to be misclassified.

In addition, the experiments were conducted on a single SMS dataset, so the results may not generalize to other domains, such as email spam or newer forms of messaging.

## How to Run
1. Clone or download this repository.
2. Open `sms_spam_classification.ipynb` in Jupyter Notebook or Google Colab.
3. Install the required Python packages.
4. Download the SMS Spam Collection Dataset and place it in the appropriate data directory.
5. Run the notebook cells sequentially from top to bottom.

## Technologies Used
- Python
- pandas
- NumPy
- NLTK
- scikit-learn
- Google Colab / Jupyter Notebook
