import pandas as pd
from pathlib import Path

SAMPLE_CSV = Path(__file__).parent.parent / "data" / "products.csv"


def load_catalog(fileobj=None):
    if fileobj is None:
        return pd.read_csv(SAMPLE_CSV)
    return pd.read_csv(fileobj)


def sample_products(df, n=5):
    return df.sample(min(n, len(df))).reset_index(drop=True)

