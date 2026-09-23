import os
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
NEW_REVIEWS = 500
TOP_K = 5
CACHE_FILE = "review_embeddings.parquet"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def read_reviews_from_snowflake():
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    query = f"""
        SELECT REVIEW_ID, CITY, RATING, COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """

    cursor = conn.cursor()
    cursor.execute(query)
    df = cursor.fetch_pandas_all()
    cursor.close()
    conn.close()

    df.columns = [col.lower() for col in df.columns]
    return df


def embed(texts):
    return embedding_model.encode(texts).tolist()


@st.cache_data
def load_reviews():
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE)

    df = read_reviews_from_snowflake()
    df = df.dropna(subset=["comment"])
    df["comment"] = df["comment"].astype(str)
    df["embedding"] = embed(df["comment"].tolist())
    df.to_parquet(CACHE_FILE, index=False)

    return df


def cosine_similarity(vec_a, vec_b):
    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)

    if denominator == 0:
        return 0

    return np.dot(vec_a, vec_b) / denominator


def find_similar_reviews(question, df):
    question_vector = embed([question])[0]

    scores = [
        cosine_similarity(question_vector, review_vector)
        for review_vector in df["embedding"]
    ]

    result = df.copy()
    result["score"] = scores

    return result.nlargest(TOP_K, "score")


def ask_llm(question, top_reviews):
    context = ""

    for _, row in top_reviews.iterrows():
        context += (
            f"({row['city']}, {row['rating']} stars) "
            f"{row['comment']}\n"
        )

    system_prompt = (
        "Answer ONLY using the customer reviews provided. "
        "Be concise. If the reviews don't cover the question, "
        "say that the provided reviews do not contain enough information."
    )

    user_prompt = f"Question: {question}\n\nReviews:\n{context}"

    response = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content


st.title("Chat with your Zomato Reviews")
st.caption(
    f"Searching {NEW_REVIEWS} reviews, answering with {CHAT_MODEL}"
)

review_df = load_reviews()

question = st.text_input(
    "Ask a question about your reviews:",
    placeholder="e.g. What are the most common complaints about delivery?"
)

if question:
    top_reviews = find_similar_reviews(question, review_df)
    answer = ask_llm(question, top_reviews)

    st.markdown("**Answer:**")
    st.write(answer)

    with st.expander("Reviews used to build this answer"):
        st.dataframe(
            top_reviews[["city", "rating", "comment"]],
            hide_index=True
        )