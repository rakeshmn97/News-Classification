# 📰 Multi-Class News Classification WebApp

A premium interactive **Streamlit** web application designed to classify news articles and headlines into semantic categories using a pipeline of **TF-IDF Vectorization** and unsupervised **K-Means Clustering**. 

Since K-Means is an unsupervised technique, this application maps each cluster to its majority category (computed dynamically from labeled training data) and provides a confidence/purity score for every prediction.

---

## 🚀 Key Features

* **🔍 Classify Article**: Paste any news headline or full article text to instantly identify its predicted category, cluster assignment, and prediction confidence.
* **📂 Batch Processing (CSV)**: Upload a CSV containing multiple articles, select the target text column, run batch classification, and download the annotated results as a new CSV.
* **📊 Cluster Explorer**: Interactive visualization showcasing the distribution of true categories across each cluster and mapping tables to evaluate prediction confidence.

---

## 🛠️ Project Structure

The project workspace consists of the following key files:

* [app.py](file:///c:/Users/USER/Desktop/vs%20aiml/Streamlit_Project/News%20Clasiffication/app.py) — The main Streamlit web application.
* [requirements.txt](file:///c:/Users/USER/Desktop/vs%20aiml/Streamlit_Project/News%20Clasiffication/requirements.txt) — Python dependencies pinned to correct versions.
* `kmeans_model.pkl` — Pre-trained K-Means clustering model.
* `tfidf_vectorizer.pkl` — Fitted TF-IDF Vectorizer.
* `news_clustered_data.pkl` — Labeled training dataset used to map clusters to human-readable categories.

---

## ⚙️ Setup & Installation

### 1. Clone & Navigate
Place the files in your project directory. Ensure that the three model/data binary files (`kmeans_model.pkl`, `tfidf_vectorizer.pkl`, and `news_clustered_data.pkl`) are placed in the root directory alongside `app.py`.

### 2. Set Up Virtual Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> [!WARNING]
> **scikit-learn version compatibility:** The serialized `.pkl` files were saved using **scikit-learn 1.6.1**. To avoid loading errors (e.g., `AttributeError` or `ModuleNotFoundError`), verify that your installed scikit-learn version matches exactly. If needed, force reinstall it using:
> ```bash
> pip install scikit-learn==1.6.1 --force-reinstall
> ```

> [!NOTE]
> Upon running the app for the first time, it will automatically download standard NLTK resource packages (`stopwords`, `wordnet`, `omw-1.4`). This requires an active internet connection on the first run, after which they are cached locally.

---

## 🏃 Run the Application

Start the Streamlit application local development server:

```bash
streamlit run app.py
```

---

## 🧠 How it Works

1. **Text Preprocessing**: The input news article undergoes a clean-up pipeline matching the training phase:
   - Stripping possessive suffix `'s`.
   - Removing all non-alphabetic characters (numbers, punctuation, symbols).
   - Lowercasing and tokenization.
   - Filtering out standard English stopwords.
   - Lemmatizing tokens to their base noun form using NLTK's `WordNetLemmatizer`.
2. **Vectorization**: Cleaned text is transformed into numerical features using the pre-fit TF-IDF vectorizer (`tfidf_vectorizer.pkl`).
3. **Clustering**: The K-Means model (`kmeans_model.pkl`) assigns the article vector to one of 8 distinct clusters.
4. **Majority-Vote Label Mapping**: The assigned cluster is matched against the labeled training data (`news_clustered_data.pkl`). The category that appears most frequently in that cluster becomes the predicted category. The ratio of that category within the cluster acts as the cluster **Confidence/Purity score**.

### ⚠️ Note on Accuracy
Because this pipeline relies on **unsupervised clustering** rather than a supervised classifier, some clusters may contain a mix of categories due to shared vocabulary (e.g., business-related terms appearing in both "Economy" and "International Relations"). Prediction confidence scores are surfaced transparently in the UI to indicate cluster purity.

