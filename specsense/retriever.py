import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

EMB_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMB_MODEL)


def build_retriever(df, use_faiss=True, use_pinecone=False):
    # minimal retriever object
    docs = df['description'].astype(str).tolist()
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)

    # embeddings
    X = embedder.encode(docs, convert_to_numpy=True)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    X = X.astype('float32')

    # FAISS index
    if use_faiss:
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
    else:
        index = None

    return {
        'df': df,
        'bm25': bm25,
        'embeddings': X,
        'faiss': index
    }


def hybrid_search(retriever, query, alpha=0.6, top_k=5, hard_constraints=None):
    df = retriever['df']
    bm25 = retriever['bm25']
    X = retriever['embeddings']
    faiss_index = retriever['faiss']

    # dense
    qv = embedder.encode([query], convert_to_numpy=True)
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
    qv = qv.astype('float32')

    D, I = faiss_index.search(qv, k=min(top_k*5, len(df)))
    dense_scores = {int(i): float(D[0][idx]) for idx, i in enumerate(I[0])}

    # sparse
    sparse_scores = bm25.get_scores(query.lower().split())

    # normalize
    def normalize(arr):
        a = np.array(arr, dtype=float)
        lo, hi = a.min(), a.max()
        rng = hi - lo if hi > lo else 1e-9
        return (a - lo) / rng

    d_norm = normalize([dense_scores.get(i, 0.0) for i in range(len(df))])
    s_norm = normalize(sparse_scores)

    fused = [(i, alpha * d_norm[i] + (1-alpha) * s_norm[i]) for i in range(len(df))]
    fused = sorted(fused, key=lambda x: x[1], reverse=True)[:top_k]

    # build explanation bundle
    results = []
    for idx, score in fused:
        row = df.iloc[idx].to_dict()
        explanation = explain_simple(row, query)
        results.append({
            'product_name': row.get('name') or row.get('title'),
            'score': round(float(score), 4),
            'price': row.get('price', ''),
            'explanation': explanation
        })
    return results


def explain_simple(row, query):
    parts = []
    parts.append(f"Matches: {row.get('title')}")
    parts.append(f"Specs: {row.get('specs', '')}")
    return "\n".join(parts)





