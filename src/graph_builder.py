from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.sparse import csr_matrix, save_npz


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

DATA_CONFIG = CONFIG["data"]
GRAPH_CONFIG = CONFIG["graph"]

DATA_DIR = BASE_DIR / DATA_CONFIG["processed_path"]

INPUT_PATH = DATA_DIR / "features.csv"

ID_COLUMN = DATA_CONFIG["id_column"]
TARGET_COLUMN = DATA_CONFIG["target_column"]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GRIDGUARD AI — GRAPH CONSTRUCTION")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

feature_columns = [
    column
    for column in df.columns
    if column not in [
        ID_COLUMN,
        TARGET_COLUMN
    ]
]

X = df[feature_columns].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median()
)


# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# SELECT K DYNAMICALLY
# ============================================================

neighbor_config = GRAPH_CONFIG[
    "neighbors"
]

min_k = neighbor_config["min_k"]
max_k = neighbor_config["max_k"]

max_allowed_k = min(
    max_k,
    len(X_scaled) - 1
)

candidate_k_values = range(
    min_k,
    max_allowed_k + 1
)

print("\nSelecting graph neighborhood size...")


# For very large datasets, evaluate a representative sample.
sample_size = min(
    len(X_scaled),
    max(1000, int(np.sqrt(len(X_scaled)) * 20))
)

rng = np.random.default_rng(
    CONFIG["project"]["random_state"]
)

sample_indices = rng.choice(
    len(X_scaled),
    size=sample_size,
    replace=False
)

X_sample = X_scaled[
    sample_indices
]


silhouette_results = []


for k in candidate_k_values:

    # +1 because nearest-neighbor search
    # includes the point itself.
    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean"
    )

    nn.fit(X_sample)

    distances, indices = nn.kneighbors(
        X_sample
    )

    # Remove self-neighbor.
    neighbor_indices = indices[:, 1:]

    # Use closest neighbor as a simple
    # behavioral cluster representation.
    labels = neighbor_indices[:, 0]

    # Silhouette requires multiple labels.
    if len(np.unique(labels)) > 1:

        score = silhouette_score(
            X_sample,
            labels,
            sample_size=min(
                len(X_sample),
                1000
            ),
            random_state=CONFIG[
                "project"
            ]["random_state"]
        )

    else:

        score = -1

    silhouette_results.append({
        "k": k,
        "silhouette_score": score
    })


selection_df = pd.DataFrame(
    silhouette_results
)

best_row = selection_df.loc[
    selection_df["silhouette_score"].idxmax()
]

selected_k = int(
    best_row["k"]
)

print(
    f"Selected K: {selected_k}"
)


selection_df.to_csv(
    DATA_DIR / "graph_k_selection.csv",
    index=False
)


# ============================================================
# BUILD FINAL GRAPH
# ============================================================

neighbors = NearestNeighbors(
    n_neighbors=selected_k + 1,
    metric="euclidean",
    n_jobs=-1
)

neighbors.fit(X_scaled)

distances, indices = neighbors.kneighbors(
    X_scaled
)


# ============================================================
# DIRECTED EDGES
# ============================================================

source_nodes = []
target_nodes = []
weights = []


for node_index in range(
    len(X_scaled)
):

    for neighbor_position in range(
        1,
        selected_k + 1
    ):

        neighbor_index = indices[
            node_index,
            neighbor_position
        ]

        distance = distances[
            node_index,
            neighbor_position
        ]

        source_nodes.append(
            node_index
        )

        target_nodes.append(
            neighbor_index
        )

        weights.append(
            1.0 / (1.0 + distance)
        )


# ============================================================
# UNDIRECTED GRAPH
# ============================================================

source_nodes = np.asarray(
    source_nodes,
    dtype=np.int64
)

target_nodes = np.asarray(
    target_nodes,
    dtype=np.int64
)

weights = np.asarray(
    weights,
    dtype=np.float32
)

reverse_source = target_nodes.copy()
reverse_target = source_nodes.copy()

source_nodes = np.concatenate([
    source_nodes,
    reverse_source
])

target_nodes = np.concatenate([
    target_nodes,
    reverse_target
])

weights = np.concatenate([
    weights,
    weights
])


# ============================================================
# REMOVE SELF LOOPS
# ============================================================

valid = (
    source_nodes != target_nodes
)

source_nodes = source_nodes[
    valid
]

target_nodes = target_nodes[
    valid
]

weights = weights[
    valid
]


# ============================================================
# REMOVE DUPLICATES
# ============================================================

edge_df = pd.DataFrame({
    "source": source_nodes,
    "target": target_nodes,
    "weight": weights
})

edge_df = edge_df.drop_duplicates(
    subset=["source", "target"]
)


# ============================================================
# SAVE EDGES
# ============================================================

edge_df.to_csv(
    DATA_DIR / "graph_edges.csv",
    index=False
)


# ============================================================
# NODE FEATURES
# ============================================================

node_features = pd.DataFrame(
    X_scaled,
    columns=feature_columns
)

node_features.insert(
    0,
    "node_index",
    np.arange(len(df))
)

node_features.insert(
    1,
    ID_COLUMN,
    df[ID_COLUMN].values
)

node_features.insert(
    2,
    TARGET_COLUMN,
    df[TARGET_COLUMN].values
)

node_features.to_csv(
    DATA_DIR / "node_features.csv",
    index=False
)


# ============================================================
# ADJACENCY MATRIX
# ============================================================

adjacency = csr_matrix(
    (
        edge_df["weight"].values,
        (
            edge_df["source"].values,
            edge_df["target"].values
        )
    ),
    shape=(
        len(df),
        len(df)
    )
)

save_npz(
    DATA_DIR / "adjacency_matrix.npz",
    adjacency
)


# ============================================================
# GRAPH SUMMARY
# ============================================================

num_nodes = len(df)

num_edges = len(edge_df)

average_degree = (
    num_edges / num_nodes
)

summary = pd.DataFrame([
    {
        "nodes": num_nodes,
        "edges": num_edges,
        "average_degree": average_degree,
        "average_edge_weight":
            edge_df["weight"].mean(),
        "minimum_edge_weight":
            edge_df["weight"].min(),
        "maximum_edge_weight":
            edge_df["weight"].max(),
        "self_loops":
            int(
                (
                    edge_df["source"]
                    ==
                    edge_df["target"]
                ).sum()
            ),
        "duplicate_edges":
            int(
                edge_df.duplicated(
                    ["source", "target"]
                ).sum()
            ),
        "selected_k":
            selected_k,
    }
])

summary.to_csv(
    DATA_DIR / "graph_summary.csv",
    index=False
)


print("\nGraph construction completed.")

print(
    summary.to_string(index=False)
)