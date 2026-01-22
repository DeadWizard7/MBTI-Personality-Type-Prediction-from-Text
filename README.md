## MBTI Personality Type Prediction from Text

MBTI Personality Type Prediction from Text is a machine learning project that classifies a user’s MBTI personality by their writing. We model each of the four MBTI dichotomies (Introversion vs. Extroversion, Intuition vs. Sensing, Thinking vs. Feeling, Judging vs. Perceiving) as a separate binary classification task. Using a dataset of ~8,600 users from PersonalityCafe forum posts (Kaggle’s “MBTI Type” dataset), we preprocess and vectorize the text and train several classical models to predict each dichotomy. The pipeline uses TF-IDF features (up to 5,000 terms) from cleaned user posts, and fits Logistic Regression, SVM, Naive Bayes, Decision Tree, and Random Forest classifiers. We evaluate performance with accuracy scores, confusion matrices, and ROC-AUC curves, and showcase results in visualizations and an interactive demo.

Installation

To run this project locally, clone the repository and install dependencies. For example:

git clone https://github.com/yourusername/mbti-text-prediction.git
cd mbti-text-prediction
pip install -r requirements.txt


The required packages include pandas, scikit-learn, imbalanced-learn, ipywidgets, and streamlit. You also need the MBTI dataset (mbti_1.csv) from Kaggle (datasnaek/mbti-type) in a data/ folder. Then you can run the Jupyter notebook for training and the Streamlit app for prediction. For example:

Launch Jupyter: jupyter notebook MBTI\ Personality\ Type\ Prediction\ from\ Text.ipynb and follow the instructions to train models or use the interactive widgets.

Launch Streamlit app: streamlit run app.py and open the web interface to input text and get an MBTI prediction.

Dataset & Preprocessing

The dataset (from Kaggle’s “MBTI Type” collection) contains 8,675 users’ last 50 PersonalityCafe forum posts each, along with their 4-letter MBTI type. We split the type into four binary targets: I/E, N/S, T/F, and J/P, encoding each as 1 or 0 (e.g. I=1 vs. E=0). Posts are cleaned by lowercasing, removing URLs/usernames and non-letter characters, then combined into a single text field. TF-IDF vectorization (with English stopwords removed) transforms the clean text into numerical feature vectors (shape ~8675×5000).

Limitations: The MBTI dataset has known biases. Class imbalance is significant: for example, far more Introverts than Extroverts in this data. The text comes from a niche forum (low domain diversity), and users’ self-reported MBTI types may contain noise. These factors limit generalization. We discuss handling imbalance (e.g. SMOTE or class weights) in our future work.

Modeling

We train separate binary classifiers for each dichotomy. Five classical models are used: Logistic Regression, Linear SVM, Multinomial Naive Bayes, Decision Tree, and Random Forest (200 trees). Models are fit on a stratified train-test split (80/20). Example code excerpt:

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X = tfidf.fit_transform(df["clean_posts"])
y = df[["IE","SN","TF","JP"]] 
# Split data and train e.g. LogisticRegression on IE:
from sklearn.linear_model import LogisticRegression
clf_IE = LogisticRegression(max_iter=500).fit(X_train, y_train["IE"])


Results (accuracy, etc.) for each dichotomy and model are reported in the notebook. We also save the trained TF-IDF and model objects for deployment (see models/).

Evaluation

We evaluate each classifier’s accuracy and visualize performance. For example, the confusion matrix and ROC curve for the Introversion–Extroversion (I/E) model illustrate class separation. Below is an example confusion matrix (I/E) and its ROC curve:

Confusion Matrix (I/E): true vs. predicted classes (0=E, 1=I).

Accuracy and AUC: We compute accuracy on the test set for each dichotomy, as well as AUC from ROC curves. A higher AUC indicates better discrimination.

Confusion Matrix: Shows true vs. predicted counts for each class (e.g., how many I’s were correctly identified). In imbalanced cases, we pay attention to minority recall.

ROC Curve: Plots True Positive Rate vs. False Positive Rate. E.g., an I/E model ROC with AUC ≈0.75 suggests decent performance.

These visual tools help diagnose errors. (Images above are illustrative; see the notebook for our actual plots.)

Interactive Demo & Deployment

We provide an interactive notebook demo using IPython widgets. Users can click “Train ML Models” to (re-)train all classifiers on the fly, and input custom text to predict the MBTI type via predict_mbti(text) (see code cells). Additionally, a simple Streamlit app (app.py) offers a web interface: users enter a text passage and get the 4-letter MBTI prediction instantly using the pretrained models. For example:

streamlit run app.py
# Then open http://localhost:8501 in a browser.

Limitations

Class Imbalance: As noted, classes are skewed (especially I/E and N/S). This can bias accuracy.

Data Diversity: Data is from one forum; language style may not generalize to other contexts (e.g. social media, blogs).

Noisy Labels: MBTI self-reports are subjective. Forum posts may not fully reflect personality.

These issues motivate careful interpretation of results and the improvements below.

Future Work (2025–2026)

We plan several enhancements to improve accuracy and robustness:

Transformer-based Models: Upgrade from TF-IDF to pretrained transformer encoders (e.g. DistilBERT or RoBERTa) to capture contextual semantics. DistilBERT, for example, is a smaller, faster BERT distilled model that achieves similar performance to larger transformers. Fine-tuning such models on the MBTI text could boost accuracy.

Sentence Embeddings: Use sentence-transformers (e.g. all-MiniLM-L6-v2) to embed posts. all-MiniLM-L6-v2 is 5× faster than larger embedding models and still high-quality. These dense vectors may improve representation over TF-IDF.

Imbalance Handling: Apply SMOTE or class-weighting to address imbalanced labels. (In prior work, SMOTE slightly improved minority-class recall.) We can also use stratified sampling and F1/AUC-based evaluation to ensure minority classes are learned.

More Data: Expand training text sources beyond PersonalityCafe. Collect MBTI-labeled posts from Reddit, Twitter, or personality quizzes, and add multilingual data to cover non-English users.

Explainability: Use SHAP or LIME to explain model predictions and identify which words or phrases are most indicative of each trait.

Ensemble Learning: Combine multiple classifiers or modalities (e.g. TF-IDF + embeddings) in ensembles or stacked models for better generalization.

Model Calibration: Improve probability estimates so confidence outputs (for ROC/AUC) are more reliable.

LLM Baselines: Evaluate few-shot/zero-shot predictions using large language models (GPT-4, etc.) as a modern baseline.

This roadmap aims to modernize the system beyond classical methods and improve its accuracy and interpretability.

Usage Example

Train Models: In the Jupyter notebook, run the cells to train classifiers. Alternatively, click the 🚀 Train ML Models button in the interactive demo to retrain on demand.

Predict via Notebook: Enter sample text in the provided input widget and click “Predict MBTI 🚀” to get a 4-letter type prediction.

Run Streamlit App: From the command line, run streamlit run app.py and open the local web app. Enter text into the box and view the predicted personality.
