from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "features.csv"

NODE_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "node_features.csv"
EDGE_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "graph_edges.csv"
ADJ_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "adjacency_matrix.npz"
SUMMARY_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "graph_summary.csv"


# ============================================================
# GRAPH SETTINGS
# ============================================================

K_NEIGHBORS = 5


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("GRIDGUARD AI - GRAPH CONSTRUCTION")
print("=" * 60)

print("\nLoading feature dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

ID_COLUMN = "CONS_NO"
TARGET_COLUMN = "FLAG"

feature_columns = [
    column
    for column in df.columns
    if column not in [ID_COLUMN, TARGET_COLUMN]
]

print(f"\nNumber of node features: {len(feature_columns)}")

print("\nNode features:")
for column in feature_columns:
    print(f"  - {column}")


# ============================================================
# PREPARE NODE FEATURES
# ============================================================

print("\nPreparing node features...")

X = df[feature_columns].copy()

# Convert all feature columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Fill missing values with feature medians
X = X.fillna(X.median())

# Safety check
if X.isnull().sum().sum() > 0:
    raise ValueError("Missing values still exist in node features.")

print(f"Node feature matrix shape: {X.shape}")


# ============================================================
# STANDARDIZE FEATURES
# ============================================================

print("\nStandardizing node features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature standardization complete.")


# ============================================================
# CREATE NODE FEATURE FILE
# ============================================================

print("\nCreating node feature table...")

node_features = pd.DataFrame(
    X_scaled,
    columns=feature_columns
)

# Add node index
node_features.insert(
    0,
    "node_index",
    np.arange(len(df))
)

# Add household ID
node_features.insert(
    1,
    ID_COLUMN,
    df[ID_COLUMN].values
)

# Add target label for later evaluation
node_features.insert(
    2,
    TARGET_COLUMN,
    df[TARGET_COLUMN].values
)

node_features.to_csv(
    NODE_OUTPUT_PATH,
    index=False
)

print(f"Saved node features to:")
print(NODE_OUTPUT_PATH)


# ============================================================
# KNN GRAPH CONSTRUCTION
# ============================================================

print("\nBuilding KNN similarity graph...")

print(f"K neighbors: {K_NEIGHBORS}")

# We request K+1 neighbors because each point is its own
# nearest neighbor and we want K actual neighboring households.
knn = NearestNeighbors(
    n_neighbors=K_NEIGHBORS + 1,
    metric="euclidean",
    n_jobs=-1
)

knn.fit(X_scaled)

distances, indices = knn.kneighbors(X_scaled)

print("KNN search complete.")


# ============================================================
# CREATE DIRECTED EDGE LIST
# ============================================================

print("\nCreating similarity edges...")

edge_list = []

for source in range(len(df)):

    for neighbor_position in range(1, K_NEIGHBORS + 1):

        target = int(indices[source, neighbor_position])

        distance = float(
            distances[source, neighbor_position]
        )

        # Convert distance into similarity.
        #
        # Smaller distance = higher similarity.
        #
        # Similarity is always positive and approaches 1
        # when the distance approaches 0.
        similarity = 1.0 / (1.0 + distance)

        edge_list.append(
            (
                source,
                target,
                similarity
            )
        )


edges_df = pd.DataFrame(
    edge_list,
    columns=[
        "source",
        "target",
        "weight"
    ]
)

print(f"Initial directed edges: {len(edges_df):,}")


# ============================================================
# REMOVE SELF-LOOPS
# ============================================================

print("\nRemoving self-loops...")

before_self_loop_removal = len(edges_df)

edges_df = edges_df[
    edges_df["source"] != edges_df["target"]
].copy()

self_loops_removed = (
    before_self_loop_removal - len(edges_df)
)

print(
    f"Self-loops removed: {self_loops_removed:,}"
)


# ============================================================
# MAKE GRAPH UNDIRECTED
# ============================================================

print("\nConverting graph to undirected...")

# Add reverse direction for every edge.
reverse_edges = edges_df.rename(
    columns={
        "source": "target",
        "target": "source"
    }
)

edges_df = pd.concat(
    [
        edges_df,
        reverse_edges
    ],
    ignore_index=True
)

print(
    f"Edges after adding reverse connections: "
    f"{len(edges_df):,}"
)


# ============================================================
# REMOVE SELF-LOOPS AGAIN
# ============================================================

# This is a safety check before deduplication.

edges_df = edges_df[
    edges_df["source"] != edges_df["target"]
].copy()


# ============================================================
# REMOVE DUPLICATE EDGES
# ============================================================

print("\nRemoving duplicate edges...")

before_duplicate_removal = len(edges_df)

edges_df = edges_df.drop_duplicates(
    subset=[
        "source",
        "target"
    ]
).reset_index(drop=True)

duplicate_edges_removed = (
    before_duplicate_removal - len(edges_df)
)

print(
    f"Duplicate edges removed: "
    f"{duplicate_edges_removed:,}"
)


# ============================================================
# FINAL SELF-LOOP CHECK
# ============================================================

self_loop_count = int(
    (
        edges_df["source"] ==
        edges_df["target"]
    ).sum()
)

duplicate_edge_count = int(
    edges_df.duplicated(
        subset=[
            "source",
            "target"
        ]
    ).sum()
)

print("\nFinal graph validation:")

print(
    f"Self-loops: {self_loop_count:,}"
)

print(
    f"Duplicate edges: {duplicate_edge_count:,}"
)


# ============================================================
# VALIDATION
# ============================================================

if self_loop_count != 0:
    raise ValueError(
        "Self-loops still exist in the graph."
    )

if duplicate_edge_count != 0:
    raise ValueError(
        "Duplicate edges still exist in the graph."
    )


# ============================================================
# SAVE EDGE LIST
# ============================================================

edges_df.to_csv(
    EDGE_OUTPUT_PATH,
    index=False
)

print("\nSaved graph edges to:")
print(EDGE_OUTPUT_PATH)


# ============================================================
# CREATE ADJACENCY MATRIX
# ============================================================

print("\nCreating sparse adjacency matrix...")

num_nodes = len(df)

adjacency_matrix = csr_matrix(
    (
        edges_df["weight"].values,
        (
            edges_df["source"].values,
            edges_df["target"].values
        )
    ),
    shape=(
        num_nodes,
        num_nodes
    )
)

save_npz(
    ADJ_OUTPUT_PATH,
    adjacency_matrix
)

print("Adjacency matrix created.")

print(
    f"Matrix shape: {adjacency_matrix.shape}"
)

print(
    f"Non-zero connections: "
    f"{adjacency_matrix.nnz:,}"
)


# ============================================================
# GRAPH STATISTICS
# ============================================================

print("\nCalculating graph statistics...")

num_edges = len(edges_df)

average_degree = (
    num_edges / num_nodes
    if num_nodes > 0
    else 0
)

average_edge_weight = (
    edges_df["weight"].mean()
    if num_edges > 0
    else 0
)

minimum_edge_weight = (
    edges_df["weight"].min()
    if num_edges > 0
    else 0
)

maximum_edge_weight = (
    edges_df["weight"].max()
    if num_edges > 0
    else 0
)


# ============================================================
# GRAPH SUMMARY
# ============================================================

graph_summary = pd.DataFrame(
    {
        "metric": [
            "nodes",
            "edges",
            "average_degree",
            "average_edge_weight",
            "minimum_edge_weight",
            "maximum_edge_weight",
            "self_loops",
            "duplicate_edges",
            "k_neighbors"
        ],
        "value": [
            num_nodes,
            num_edges,
            average_degree,
            average_edge_weight,
            minimum_edge_weight,
            maximum_edge_weight,
            self_loop_count,
            duplicate_edge_count,
            K_NEIGHBORS
        ]
    }
)

graph_summary.to_csv(
    SUMMARY_OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("GRAPH CONSTRUCTION COMPLETE")
print("=" * 60)

print(f"\nNodes:              {num_nodes:,}")
print(f"Edges:              {num_edges:,}")
print(f"Average degree:     {average_degree:.4f}")
print(f"Average weight:     {average_edge_weight:.6f}")
print(f"Minimum weight:     {minimum_edge_weight:.6f}")
print(f"Maximum weight:     {maximum_edge_weight:.6f}")
print(f"Self-loops:         {self_loop_count}")
print(f"Duplicate edges:    {duplicate_edge_count}")
print(f"K neighbors:        {K_NEIGHBORS}")

print("\nFiles created:")

print(f"1. {NODE_OUTPUT_PATH}")
print(f"2. {EDGE_OUTPUT_PATH}")
print(f"3. {ADJ_OUTPUT_PATH}")
print(f"4. {SUMMARY_OUTPUT_PATH}")

print("\nGraph is ready for Phase 8.")