"""
Data prep for the corpus map (notebooks/map.py): one vector per paper, a 2D UMAP
projection of them, and labelled clusters ("islands"). Read-only on papers.duckdb.

A paper's vector is the mean of its chunk embeddings (boilerplate has none), re-normalised.
The mean is accumulated batch by batch from DuckDB's Arrow stream, so the ~345K x 1024
chunk matrix is never held at once. Vectors and projection are cached in
data/paper_map.npz and rebuilt only when the number of embeddings for the model changes
(or with refresh=True), so reopening the notebook is instant.

Needs numpy, scikit-learn and umap-learn: the `browse` pixi environment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb
import numpy as np

import config

CACHE_PATH = config.DATA_DIR / 'paper_map.npz'
BATCH_ROWS = 50_000
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.05  # tight islands with open water between them
MIN_ISLAND_PAPERS = 120
LABEL_TERMS = 3

# Words that say nothing about which island a paper belongs on.
GENERIC_TERMS = {
    'study', 'studies', 'analysis', 'using', 'based', 'new', 'observations',
    'observational', 'observed', 'data', 'results', 'approach', 'method',
    'methods', 'model', 'models', 'case', 'evidence', 'properties', 'high',
    'low', 'large', 'survey', 'implications', 'role', 'search', 'constraints',
}  # fmt: skip


@dataclass
class PaperMap:
    bibcodes: list[str]
    vectors: np.ndarray  # (n, d) float32, unit length
    xy: np.ndarray  # (n, 2) float32
    key: str


def _cache_key(con: duckdb.DuckDBPyConnection, model: str) -> str:
    n_emb = con.execute(
        'SELECT count(*) FROM chunk_embeddings WHERE model = ?', [model]
    ).fetchall()[0][0]
    return f'{model}|{n_emb}|{UMAP_NEIGHBORS}|{UMAP_MIN_DIST}'


def paper_vectors(
    con: duckdb.DuckDBPyConnection, model: str
) -> tuple[list[str], np.ndarray]:
    """Mean chunk embedding per paper, unit length. Papers without embeddings are absent."""
    bibcodes = [
        r[0]
        for r in con.execute(
            'SELECT DISTINCT bibcode FROM chunk_embeddings WHERE model = ? ORDER BY 1',
            [model],
        ).fetchall()
    ]
    index = {b: i for i, b in enumerate(bibcodes)}
    sums: np.ndarray | None = None

    reader = (
        con.cursor()
        .execute(
            'SELECT bibcode, embedding FROM chunk_embeddings WHERE model = ? ORDER BY bibcode',
            [model],
        )
        .to_arrow_reader(BATCH_ROWS)
    )
    for batch in reader:
        if batch.num_rows == 0:
            continue
        col = batch.column('embedding')
        dim = len(col[0])
        vecs = col.flatten().to_numpy(zero_copy_only=False).reshape(-1, dim)
        ids = np.fromiter(
            (index[b] for b in batch.column('bibcode').to_pylist()),
            dtype=np.int64,
            count=batch.num_rows,
        )
        if sums is None:
            sums = np.zeros((len(bibcodes), dim), dtype=np.float32)
        # Rows arrive sorted by bibcode, so equal ids are contiguous: reduce each run,
        # then add (a bibcode may continue in the next batch, hence += not =). The
        # sums are normalised below, so no counts are needed.
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        sums[ids[starts]] += np.add.reduceat(vecs, starts, axis=0)

    assert sums is not None, f'no embeddings for {model}'
    sums /= np.linalg.norm(sums, axis=1, keepdims=True)
    return bibcodes, sums


def project(vectors: np.ndarray) -> np.ndarray:
    import umap

    return umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric='cosine',
        random_state=0,  # reproducible layout; fine at this size
    ).fit_transform(vectors).astype(np.float32)  # fmt: skip


def load(
    con: duckdb.DuckDBPyConnection,
    model: str = config.EMBED_MODEL,
    refresh: bool = False,
) -> PaperMap:
    key = _cache_key(con, model)
    if not refresh and CACHE_PATH.exists():
        with np.load(CACHE_PATH, allow_pickle=False) as z:
            if str(z['key']) == key:
                return PaperMap(list(z['bibcodes']), z['vectors'], z['xy'], key)
    bibcodes, vectors = paper_vectors(con, model)
    xy = project(vectors)
    np.savez(CACHE_PATH, key=key, bibcodes=np.array(bibcodes), vectors=vectors, xy=xy)
    return PaperMap(bibcodes, vectors, xy, key)


def find_islands(xy: np.ndarray) -> np.ndarray:
    """Cluster id per paper from density clustering of the 2D layout; -1 = open water."""
    from sklearn.cluster import HDBSCAN

    return HDBSCAN(
        min_cluster_size=MIN_ISLAND_PAPERS, min_samples=10, copy=True
    ).fit_predict(xy)


def name_islands(
    cluster: np.ndarray, titles: list[str], keywords: list[list[str]]
) -> dict[int, str]:
    """Name each island by its most distinctive title/keyword terms (c-TF-IDF)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

    ids = sorted(set(cluster.tolist()) - {-1})
    if not ids:
        return {}
    docs = []
    for c in ids:
        members = np.flatnonzero(cluster == c)
        docs.append(
            ' '.join(
                (titles[i] or '') + ' ' + ' '.join(keywords[i] or []) for i in members
            ).lower()
        )
    vec = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS | GENERIC_TERMS),
        token_pattern=r'(?u)\b[a-z][a-z\-]{2,}\b',
        ngram_range=(1, 2),
        min_df=1,
    )
    tf = vec.fit_transform(docs).toarray().astype(np.float64)
    tf /= tf.sum(axis=1, keepdims=True) + 1e-9
    # Terms found on every island ("astrophysics") score ~0; ones confined to a few win.
    n_islands_with = (tf > 0).sum(axis=0)
    scores = tf * np.log((len(ids) + 1) / n_islands_with)
    terms = np.array(vec.get_feature_names_out())
    names = {}
    for row, c in zip(scores, ids):
        picked: list[str] = []
        for j in np.argsort(row)[::-1]:
            t = terms[j]
            # Skip a term already covered by a picked one ("red" after "little red dots").
            if any(t in p or p in t for p in picked):
                continue
            picked.append(t)
            if len(picked) == LABEL_TERMS:
                break
        names[c] = re.sub(r'\s+', ' ', ' · '.join(picked))
    return names
