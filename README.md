# SpecSense AI

SpecSense AI is an **AI-powered product specification understanding and recommendation system**. Instead of relying on rigid filters or past user behavior, SpecSense AI interprets **free-form natural language queries**, extracts user intent, and matches it with structured product data to return **context-aware, explainable product recommendations**.

This solution is designed for enterprises with existing product catalogs who want to offer **human-like, conversational product matching** without rebuilding their recommendation engines from scratch.

## 🧠 Problem Statement

Traditional recommender systems often fall short because they:

- Require users to click through **filters and menus**
- Lack understanding of **vague or contextual requests**
- Provide **little transparency** about *why* recommendations are made
- Struggle to combine:
  - **Hard constraints** (price, RAM, weight, storage)
  - **Soft preferences** (battery life, “good for coding”, travel-friendly)
  - **Context** (budget, lifestyle, usage intent)

This leads to:
- Customer frustration or decision fatigue
- Drop-offs during product discovery
- Underutilization of product diversity in catalogs

## 🎯 What SpecSense AI Does

SpecSense AI builds an **intelligence layer** over your product catalog:

1. Understands free-form user intent.
2. Extracts structured constraints and soft preferences.
3. Matches user intent with structured catalog data.
4. Suggests closest alternatives when no perfect match exists.

## 🗂 Data Inputs

| Data Type | Format | Purpose |
|----------|--------|---------|
| **Product Catalog** | CSV / JSON | Names, descriptions, specifications |
| **User Query** | Natural language text | Expresses needs and context |

## 🗂 Project Structure

SpecSense-AI/
├── app.py

├── requirements.txt

├── .env

├── data/

│   └── products.csv

├── specsense/

│   ├── extractor.py

│   ├── retriever.py

│   ├── ranker.py

│   ├── utils.py

│   └── __init__.py

└── tests/

    └── test_pipeline.py

## 🚀 Run

pip install -r requirements.txt
python app.py

## 🧩 Core Modules

### `extractor.py`
- Identifies hard constraints (price, size, features)
- Detects soft preferences (comfort, reliability, travel use)

### `retriever.py`
- Converts products into embedding vectors
- Performs semantic similarity search

### `ranker.py`
- Scores retrieved candidates based on:
  - Query-spec alignment
  - Feature coverage
  - Soft preference match strength


🛠 Technical Requirements

Python 3.12+

Pandas

python-dotenv

(Optional) OpenAI / Gemini API

Streamlit (if UI mode is enabled)
