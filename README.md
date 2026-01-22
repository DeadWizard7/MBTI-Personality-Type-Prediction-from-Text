#MBTI Personality Type Prediction from Text

A machine learning project that predicts Myers-Briggs Type Indicator (MBTI) personality types (16 types) from users' written text (mainly social media posts).
The system classifies each of the four dichotomies separately:

I/E (Introversion vs Extraversion)
N/S (Intuition vs Sensing)
F/T (Feeling vs Thinking)
J/P (Judging vs Perceiving)

Final MBTI type is built by combining the four independent predictions (e.g. INFP, ESTJ).
Main Features

Text preprocessing: cleaning (URLs, mentions, special characters), stopword removal, lemmatization
Feature extraction: TF-IDF vectorization
Multiple classical ML models per dichotomy:
Logistic Regression
Linear SVM
Multinomial Naive Bayes
Decision Tree
Random Forest (200 trees)

Separate binary classification for each MBTI dimension
Simple prediction function that returns full 4-letter MBTI type
Model comparison (accuracy, confusion matrix, ROC-AUC visualized for selected dimensions)
Interactive prediction demo using ipywidgets (Jupyter)
Basic Streamlit app skeleton for deployment

Main Dataset Limitations

Heavily imbalanced classes (especially strong imbalance in I/E and N/S dimensions — introverts and intuitives are significantly over-represented)
Data mostly comes from personalitycafe forum posts → very specific writing style, topics, and community language
Short / noisy / low-quality posts in many cases
No demographic diversity control (age, gender, culture, native language)
Limited size compared to modern LLMs datasets (usually ~8k–20k labeled examples depending on source)
Labels are self-reported → noise from mistyping or misunderstanding MBTI

Recommended Improvements (2025–2026 perspective)
Must-do / high-impact:

Handle class imbalance properly (SMOTE, class_weight, focal loss, undersampling majority + oversampling minority)
Try modern transformer models instead of TF-IDF + classical ML
DistilBERT / RoBERTa / DeBERTa-v3 / ELECTRA / TinyBERT
Fine-tune per dichotomy or multi-label head for all four at once

Use sentence transformers (all-MiniLM-L6-v2, etc.) + simple classifier on top → often better than TF-IDF
Add data augmentation (back-translation, synonym replacement, random deletion)
Collect / mix more diverse data sources (Reddit, Twitter/X, Tumblr personality communities)
Evaluate with proper stratified k-fold cross-validation
Add confidence scores / probability calibration for predictions
Create model ensemble (voting or stacking the best 2–3 models per dichotomy)

Nice-to-have:
Multilingual support (mBERT, XLM-R)
Explainability (SHAP / LIME on important words per prediction)
Web demo with Gradio or Streamlit + caching
Compare against zero-shot / few-shot prompting with modern LLMs (Llama-3, Mistral, Gemma-2, Qwen-2)
