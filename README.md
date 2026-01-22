## MBTI Personality Type Prediction from Text

A machine learning project that predicts Myers–Briggs Type Indicator (MBTI) personality types from users’ written text (primarily social media posts).

The system predicts each MBTI dichotomy independently and combines them to produce a final 4-letter MBTI type (e.g., INFP, ESTJ).

🔍 Problem Overview

The MBTI framework consists of four binary personality dimensions:

I / E — Introversion vs Extraversion

N / S — Intuition vs Sensing

F / T — Feeling vs Thinking

J / P — Judging vs Perceiving

Instead of treating MBTI as a single 16-class problem, this project models it as four independent binary classification tasks, which improves flexibility, interpretability, and model performance.

🧪 Project Pipeline
1️⃣ Text Preprocessing

URL and mention removal

Special character and digit cleaning

Lowercasing

Stopword removal

Lemmatization

2️⃣ Feature Extraction

TF-IDF vectorization

Word-level n-grams

3️⃣ Modeling (Per Dichotomy)

Each MBTI dimension is trained separately using:

Logistic Regression

Linear Support Vector Machine (SVM)

Multinomial Naive Bayes

Decision Tree

Random Forest (200 trees)

4️⃣ Final MBTI Prediction

Predictions from the four binary classifiers are combined into a single MBTI type:

(I/E) + (N/S) + (F/T) + (J/P) → Final MBTI

📊 Evaluation & Visualization

Accuracy comparison across models

Confusion matrices

ROC–AUC curves (for selected dimensions)

Stratified train/test splits

🧩 Interactive Components

Prediction function returning the full MBTI type from raw text

Jupyter Notebook demo using ipywidgets

Streamlit app skeleton for deployment

⚠️ Dataset Limitations

Severe class imbalance, especially in I/E and N/S

Data sourced mainly from PersonalityCafe forums

Writing style and topics are community-specific

Short, noisy, and low-quality posts

No demographic diversity control

Limited dataset size (~8k–20k samples)

Labels are self-reported, introducing noise and mistyping

🚀 Recommended Improvements (2025–2026)
✅ Must-Do / High-Impact

Proper imbalance handling:

class_weight

SMOTE / undersampling

Focal loss

Replace TF-IDF with transformer-based models:

DistilBERT

RoBERTa

DeBERTa-v3

ELECTRA

Sentence embeddings:

all-MiniLM-L6-v2

Fine-tuning:

Per-dichotomy models or

Single multi-label transformer head

Data augmentation:

Back-translation

Synonym replacement

Random deletion

Diverse data sources:

Reddit

Twitter/X

Tumblr

Stratified k-fold cross-validation

Probability calibration + confidence scores

Model ensembling (voting / stacking)

✨ Nice-to-Have

Multilingual support (mBERT, XLM-R)

Explainability (SHAP / LIME)

Gradio or Streamlit web demo

Comparison with modern LLM zero-shot / few-shot prompting:

LLaMA-3

Mistral

Gemma-2

Qwen-2
