# 🧠 Sentiment Analysis of Customer Feedback Using NLP

![NLP](https://img.shields.io/badge/NLP-Sentiment--Analysis-blue)
![Python](https://img.shields.io/badge/Python-3.8+-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview

This project aims to perform **Sentiment Analysis** on restaurant reviews using Natural Language Processing (NLP). The goal is to classify whether a review is **positive** or **negative** based on its textual content.

We experimented with multiple preprocessing techniques and classification models including:
- **Lemmatization** for text normalization
- **TF-IDF Vectorization** for feature extraction
- **Multinomial Naive Bayes** and **Random Forest** for model building

---

## 🍽️ Why Sentiment Analysis?

Sentiment analysis helps businesses:
- Gauge customer satisfaction
- Identify negative trends early
- Make informed, data-driven decisions

This project shows how to build such a system using a labeled dataset of restaurant reviews.

---

## 📂 Dataset

- **Size:** 1000 reviews
- **Features:** Text review and label (1 = liked, 0 = not liked)
- **Source:** [Mention the dataset source or upload `Restaurant_Reviews.tsv`]

---

## 🛠️ Tech Stack

- **Languages:** Python
- **Libraries:**
  - `nltk`
  - `scikit-learn`
  - `pandas`
  - `re`
  - `seaborn`
- **Algorithms Used:**
  - Multinomial Naive Bayes
  - Random Forest Classifier

---

## ⚙️ NLP Pipeline

1. **Text Cleaning:**
   - Remove non-alphabetic characters
   - Convert to lowercase
   - Remove stopwords
   - Lemmatize words

2. **Feature Extraction:**
   - Applied `TF-IDF Vectorizer` to convert text into numerical vectors

3. **Model Building:**
   - **Multinomial Naive Bayes:** Simple, fast, but lower accuracy (~77%)
   - **Random Forest Classifier:** Better accuracy (~81.11%), captures complex patterns

---

## 🔍 Evaluation Metrics

- **Accuracy Score**
- **Confusion Matrix (visualized using Seaborn)**

| Model                 | Accuracy |
|----------------------|----------|
| Naive Bayes          | ~77%     |
| Random Forest        | **81.11%**   |

---

## ⚠️ Challenges Faced

- Choosing between **stemming vs. lemmatization**
- Deciding **CountVectorizer vs. TF-IDF**
- Avoiding **overfitting** in Random Forest
- **Imbalanced dataset** (slightly more positive reviews)
- Extensive **hyperparameter tuning**

---

## 📈 Results

✅ **Best Performing Model:** Random Forest Classifier  
📊 **Achieved Accuracy:** 81.11%  
🧠 **Insight:** TF-IDF with ensemble models can significantly enhance text classification tasks.

---

## 🔗 How to Run

**1. Clone the repo:**
   git clone https://github.com/your-username/sentiment-analysis-nlp.git
   cd sentiment-analysis-nlp
**2. Install dependencies:**
           pip install -r requirements.txt
**3. Run the Flask App**
      python app.py
      Visit: http://localhost:5000
