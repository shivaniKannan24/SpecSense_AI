import streamlit as st
from specsense.utils import load_catalog, sample_products
from specsense.extractor import extract_constraints
from specsense.retriever import build_retriever, hybrid_search
from specsense.ranker import explain_matches
import os

st.set_page_config(page_title="SpecSense AI — Product Matcher", layout="wide")

st.title("SpecSense AI — Explainable Product Matching")

# Sidebar: Upload dataset / config
with st.sidebar:
    st.header("Dataset & Config")
    uploaded = st.file_uploader("Upload product catalog (.csv)", type=["csv"])
    if uploaded:
        df = load_catalog(uploaded)
    else:
        st.info("Using sample product catalog. Upload to replace.")
        df = load_catalog(None)

    st.write(f"Products: {len(df)}")
    st.markdown("---")
    st.header("Model & Backend")
    use_faiss = st.checkbox("Use local FAISS (CPU)", value=True)
    use_pinecone = st.checkbox("Use Pinecone (if configured)", value=False)

# Build retriever once
retriever = build_retriever(df, use_faiss=use_faiss, use_pinecone=use_pinecone)

st.subheader("Describe what you want")
query = st.text_area(
    "Natural language product preference",
    height=120,
    placeholder="e.g., Lightweight laptop under 30k INR, good for coding and long battery life"
)

# Controls aligned horizontally below search bar
c1, c2, c3 = st.columns([1,1,2])
with c1:
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)
with c2:
    alpha = st.slider("Hybrid alpha", 0.0, 1.0, 0.6)
with c3:
    find_button = st.button("🔍 Find Matches", use_container_width=True)

# 🟩 Results appear RIGHT BELOW the search bar
if find_button:
    if not query.strip():
        st.warning("Please type a query.")
    else:
        with st.spinner("Extracting constraints..."):
            constraints = extract_constraints(query)

        with st.spinner("Searching catalog..."):
            matches = hybrid_search(
                retriever,
                query,
                alpha=alpha,
                top_k=top_k,
                hard_constraints=constraints.get("hard", {})
            )

        st.write("### ⭐ Best Matches")
       
        # How many cards per row
        cards_per_row = 3

        for i in range(0, len(matches), cards_per_row):
            row_matches = matches[i:i+cards_per_row]
            cols = st.columns(cards_per_row)

            for col, m in zip(cols, row_matches):
                with col:
                    st.markdown(f"""
                    <div style="padding:15px; border-radius:12px; border:1px solid #DDD; background:#FAFAFA;">
                        <h4 style="margin-bottom:5px;">{m['product_name']}</h4>
                        <p><b>Price:</b> ₹{m['price']}</p>
                        <p><b>Score:</b> {round(m['score'], 3)}</p>
                    </div>
                    """, unsafe_allow_html=True)

