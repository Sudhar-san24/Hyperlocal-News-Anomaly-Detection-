# 📰 Hyperlocal News Anomaly Detection 🔍

A Machine-Learning powered Streamlit application that analyzes news articles and detects anomalies using **Zero-Shot Classification**, **Sentiment Analysis**, **TF-IDF Similarity**, and **Named Entity Recognition (NER)**.

---

## 🚀 Project Overview

Fake and manipulated news content is increasing rapidly — especially on hyperlocal media platforms. This application helps detect:

✔ News category
✔ Sentiment tone
✔ Location extraction (NER)
✔ Anomaly score based on similarity with trusted dataset
✔ Confidence score from transformer model
✔ Final credibility status

The system provides a **visual dashboard** showing insights in a clean user interface.

---

## 🧠 Features

| Feature                          | Description                                              |
| -------------------------------- | -------------------------------------------------------- |
| 🏷 Zero-Shot News Classification | Assigns news category using a transformer model          |
| 💬 Sentiment Analysis            | Detects tone as Positive, Negative, or Neutral           |
| 📍 Location Extraction           | Extracts city/state names using SpaCy                    |
| 📊 Similarity Check              | TF-IDF based cosine similarity to detect unusual content |
| 🚨 Anomaly Score                 | Predicts how unusual or fake the content might be        |
| 🎨 UI Dashboard                  | Built with Streamlit + modern glassmorphism UI           |

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Streamlit**
* **Transformers (HuggingFace)**
* **VADER Sentiment Analyzer**
* **SpaCy (en_core_web_sm)**
* **Scikit-Learn (TF-IDF, similarity scoring)**

---

## 📂 Folder Structure

```
Hyperlocal-News-Anomaly-Detection/
│── src/
│   ├── app.py
│   ├── preprocessing.py
│── data/
│   └── cleaned_final_dataset.csv (ignored in repo)
│── models/ (ignored in repo)
│── README.md
│── requirements.txt
│── .gitignore
```

---

## 📥 Installation and Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Sudhar-san24/Hyperlocal-News-Anomaly-Detection-.git
cd Hyperlocal-News-Anomaly-Detection-
```

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

Windows:

```bash
.venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download SpaCy model

```bash
python -m spacy download en_core_web_sm
```

---

## ▶️ Run the Application

```bash
streamlit run src/app.py
```

---

## 🧩 How It Works

1. User enters a news article
2. HuggingFace model predicts category & confidence
3. VADER calculates sentiment
4. SpaCy extracts location entities
5. TF-IDF similarity is computed against the dataset
6. Anomaly score is generated
7. System labels the article as:

| Anomaly Score | Status                               |
| ------------- | ------------------------------------ |
| 0.0 – 0.3     | 🟢 Normal Verified                   |
| 0.31 – 0.6    | 🟡 Rare — Review Required            |
| 0.61 – 0.85   | 🔵 Unusual — Low Familiarity         |
| 0.86 – 1.0    | 🔴 Highly Deviating — Potential Fake |

---

## 📌 Future Enhancements

* 🧬 Fine-tuned transformer model
* 🌍 Multi-language support (Tamil, Hindi, Telugu)
* 🧾 Export results to PDF
* 🧠 Model deployment to HuggingFace Hub
* ☁ Deployment on Streamlit Cloud or Docker

---

## 📄 License

This project is licensed under the **MIT License** — free to use and modify.

---

## 👨‍💻 Author

**Sudharsan Udhayakumar**
💼 Data Scientist / ML Developer
📧 [ssudhar525@gmail.com)




