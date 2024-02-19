# Spam Detection & Emotion Analysis with DL models

Struggling with spam or deciphering online sentiment? This repository delves into text classification using two powerful deep learning techniques: **LSTM** and fine-tuned **DistilBERT**. Explore how these models tackle real-world tasks like **detecting unwanted emails** and **identifying emotions in tweets**.


## Installation

1. Clone the repository:

```bash
git clone https://github.com/hoverslam/nlp-text-classification/
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebooks: 
    - [spam_detection](https://github.com/hoverslam/nlp-text-classification/blob/main/notebooks/spam_detection.ipynb)
    - [emotion_analysis](https://github.com/hoverslam/nlp-text-classification/blob/main/notebooks/emotion_analysis.ipynb)


## Results

### Spam detection

|            | Accuracy   | Precision* | Recall*    | F1-score*  |
|------------|:----------:|:----------:|:----------:|:----------:|
| LSTM       | 0.9559     | 0.9541     | 0.9545     | 0.9543     |
| DistilBERT | **0.9886** | **0.9886** | **0.9877** | **0.9882** |


### Emotion analysis

|            | Accuracy   | Precision* | Recall*    | F1-score*  |
|------------|:----------:|:----------:|:----------:|:----------:|
| LSTM       | 0.8643     | 0.8242     | 0.8253     | 0.8241     |
| DistilBERT | **0.9260** | **0.8912** | **0.8879** | **0.8891** |

\* macro-average


## License

The code in this project is licensed under the [MIT License](LICENSE.txt).