import re
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="centered",
)

# ---------------------------------------------------------
# One-time NLTK data download (cached)
# ---------------------------------------------------------
@st.cache_resource
def setup_nltk():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

STOP_WORDS, LEMMATIZER = setup_nltk()

# ---------------------------------------------------------
# Load model, vectorizer, and labeled cluster data
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    kmeans = joblib.load("kmeans_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    data = joblib.load("news_clustered_data.pkl")
    return kmeans, tfidf, data

try:
    kmeans, tfidf, news_data = load_artifacts()
    load_error = None
except Exception as e:
    kmeans, tfidf, news_data = None, None, None
    load_error = str(e)


@st.cache_data
def build_cluster_category_map(_news_data: pd.DataFrame):
    """Map each KMeans cluster to its majority news Category, plus a purity score."""
    ct = pd.crosstab(_news_data["cluster"], _news_data["Category"])
    majority = ct.idxmax(axis=1)
    purity = (ct.max(axis=1) / ct.sum(axis=1)).round(4)
    return majority.to_dict(), purity.to_dict(), ct


# ---------------------------------------------------------
# Text cleaning — mirrors the preprocessing used to build
# `cleaned_news` in the training data:
#   1. strip possessive "'s"
#   2. remove all non-letter characters
#   3. lowercase + tokenize on whitespace
#   4. remove stopwords
#   5. lemmatize (noun form)
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("📰 Multi-Class News Classifier")
st.write(
    "Classify a news article into one of several categories using a "
    "TF-IDF + KMeans clustering pipeline. Each cluster is mapped to its "
    "most common category from labeled training articles."
)

if load_error:
    st.error(
        "⚠️ Could not load the model files. This usually means a "
        "scikit-learn version mismatch between the environment that "
        "trained the model and the one running this app.\n\n"
        f"Details: {load_error}\n\n"
        "Fix: make sure the scikit-learn version matches (see requirements.txt / README)."
    )
    st.stop()

cluster_to_category, cluster_purity, crosstab = build_cluster_category_map(news_data)

st.divider()

tab_single, tab_batch, tab_explore = st.tabs(["🔍 Classify Article", "📂 Batch (CSV)", "📊 Explore Clusters"])

# ---------------------------------------------------------
# TAB 1: Single article classification
# ---------------------------------------------------------
with tab_single:
    article_text = st.text_area(
        "Paste a news article or headline",
        height=200,
        placeholder="e.g. The central bank raised interest rates by 0.5% today, citing persistent inflation concerns...",
    )

    if st.button("🧠 Classify", type="primary", use_container_width=True):
        if not article_text.strip():
            st.warning("Please enter some article text first.")
        else:
            cleaned = clean_text(article_text)
            if not cleaned.strip():
                st.warning("After cleaning, no usable text remained — try a longer article.")
            else:
                X = tfidf.transform([cleaned])
                cluster_id = int(kmeans.predict(X)[0])
                category = cluster_to_category.get(cluster_id, "Unknown")
                purity = cluster_purity.get(cluster_id, 0.0)

                st.success("Classification complete!")
                col1, col2 = st.columns(2)
                col1.metric("Predicted Category", category)
                col2.metric("Cluster Confidence", f"{purity:.0%}")

                st.caption(
                    f"Assigned to cluster **#{cluster_id}** — {purity:.0%} of labeled "
                    f"training articles in this cluster belong to **{category}**."
                )

                with st.expander("See cleaned text used for prediction"):
                    st.code(cleaned, language=None)

                with st.expander(f"Sample articles from cluster #{cluster_id}"):
                    samples = news_data[news_data["cluster"] == cluster_id][["News", "Category"]].head(5)
                    for _, row in samples.iterrows():
                        st.markdown(f"**[{row['Category']}]** {row['News']}")

# ---------------------------------------------------------
# TAB 2: Batch classification via CSV upload
# ---------------------------------------------------------
with tab_batch:
    st.write("Upload a CSV with a column of article text to classify multiple articles at once.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())

        text_col = st.selectbox("Which column contains the article text?", df.columns.tolist())

        if st.button("Classify all rows", type="primary"):
            with st.spinner("Classifying..."):
                cleaned_texts = df[text_col].astype(str).apply(clean_text)
                X = tfidf.transform(cleaned_texts)
                clusters = kmeans.predict(X)
                df["predicted_cluster"] = clusters
                df["predicted_category"] = [cluster_to_category.get(int(c), "Unknown") for c in clusters]
                df["cluster_confidence"] = [cluster_purity.get(int(c), 0.0) for c in clusters]

            st.success(f"Classified {len(df)} articles.")
            st.dataframe(df)

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results as CSV",
                data=csv_out,
                file_name="classified_news.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------
# TAB 3: Explore clusters
# ---------------------------------------------------------
with tab_explore:
    st.write("How the training articles' true categories distribute across each KMeans cluster:")

    fig, ax = plt.subplots(figsize=(8, 5))
    crosstab.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Articles")
    ax.set_title("Category Distribution per Cluster")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

    st.write("Cluster → majority category mapping used for predictions:")
    mapping_df = pd.DataFrame(
        {
            "Cluster": list(cluster_to_category.keys()),
            "Majority Category": list(cluster_to_category.values()),
            "Purity (Confidence)": [f"{cluster_purity[c]:.0%}" for c in cluster_to_category.keys()],
        }
    ).sort_values("Cluster")
    st.dataframe(mapping_df, use_container_width=True, hide_index=True)

    st.info(
        "Note: since this pipeline is built on **unsupervised clustering** (not a "
        "supervised classifier), low-purity clusters mean predictions for articles "
        "landing there are less reliable. Clusters mix categories when articles "
        "share similar vocabulary (e.g. economic language appearing in both "
        "'Economy' and 'International relations' articles)."
    )

st.divider()
st.caption(
    "Pipeline: TF-IDF vectorizer → KMeans clustering → majority-vote category mapping. "
    "Predictions are estimates based on unsupervised clustering, not a trained classifier."
)
