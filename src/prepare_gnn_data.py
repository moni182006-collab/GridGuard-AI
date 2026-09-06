from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.data import Data


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

NODE_PATH = BASE_DIR / "data" / "processed" / "node_features.csv"
EDGE_PATH = BASE_DIR / "data" / "processed" / "graph_edges.csv"

OUTPUT_PATH = BASE_DIR / "data" / "processed" / "gnn_data.pt"

GRAPH_IMAGE_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "graph_sample.png"
)

GRAPH_STATS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "gnn_graph_summary.csv"
)


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

GRAPH_SAMPLE_NODES = 500


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("GRIDGUARD AI - PYTORCH GEOMETRIC DATA PREPARATION")
print("=" * 70)


# ============================================================
# LOAD NODE DATA
# ============================================================

print("\n[1/8] Loading node features...")

node_df = pd.read_csv(NODE_PATH)

print(f"Node table shape: {node_df.shape}")

required_node_columns = [
    "node_index",
    "CONS_NO",
    "FLAG"
]

for column in required_node_columns:
    if column not in node_df.columns:
        raise ValueError(
            f"Required column missing from node_features.csv: {column}"
        )


# ============================================================
# IDENTIFY NODE FEATURES
# ============================================================

metadata_columns = [
    "node_index",
    "CONS_NO",
    "FLAG"
]

feature_columns = [
    column
    for column in node_df.columns
    if column not in metadata_columns
]

print(f"Number of node features: {len(feature_columns)}")

print("\nFeatures:")
for feature in feature_columns:
    print(f"  - {feature}")


# ============================================================
# VALIDATE NODE INDEX
# ============================================================

print("\nValidating node indices...")

expected_indices = np.arange(len(node_df))

actual_indices = node_df["node_index"].to_numpy()

if not np.array_equal(
    expected_indices,
    actual_indices
):
    raise ValueError(
        "node_index values are not sequential."
    )

print("Node indices are valid.")


# ============================================================
# CREATE NODE FEATURE TENSOR
# ============================================================

print("\n[2/8] Creating node feature tensor...")

x_numpy = node_df[
    feature_columns
].to_numpy(
    dtype=np.float32
)

x = torch.tensor(
    x_numpy,
    dtype=torch.float32
)

print(f"x shape: {x.shape}")
print(f"x dtype: {x.dtype}")


# ============================================================
# CREATE LABEL TENSOR
# ============================================================

print("\n[3/8] Creating label tensor...")

y_numpy = node_df[
    "FLAG"
].to_numpy(
    dtype=np.int64
)

y = torch.tensor(
    y_numpy,
    dtype=torch.long
)

print(f"y shape: {y.shape}")

print("\nLabel distribution:")

unique_labels, label_counts = np.unique(
    y_numpy,
    return_counts=True
)

for label, count in zip(
    unique_labels,
    label_counts
):
    print(
        f"  FLAG={label}: {count:,}"
    )


# ============================================================
# LOAD EDGE DATA
# ============================================================

print("\n[4/8] Loading graph edges...")

edge_df = pd.read_csv(EDGE_PATH)

print(
    f"Edge table shape: {edge_df.shape}"
)

required_edge_columns = [
    "source",
    "target",
    "weight"
]

for column in required_edge_columns:
    if column not in edge_df.columns:
        raise ValueError(
            f"Required column missing from graph_edges.csv: {column}"
        )


# ============================================================
# VALIDATE EDGES
# ============================================================

print("\nValidating edges...")

num_nodes = len(node_df)

if edge_df["source"].min() < 0:
    raise ValueError("Negative source node index found.")

if edge_df["target"].min() < 0:
    raise ValueError("Negative target node index found.")

if edge_df["source"].max() >= num_nodes:
    raise ValueError(
        "Source node index exceeds number of nodes."
    )

if edge_df["target"].max() >= num_nodes:
    raise ValueError(
        "Target node index exceeds number of nodes."
    )


# Check self-loops
self_loops = (
    edge_df["source"] ==
    edge_df["target"]
).sum()

if self_loops != 0:
    raise ValueError(
        f"Graph contains {self_loops} self-loops."
    )


# Check duplicate directed edges
duplicate_edges = edge_df.duplicated(
    subset=[
        "source",
        "target"
    ]
).sum()

if duplicate_edges != 0:
    raise ValueError(
        f"Graph contains {duplicate_edges} duplicate edges."
    )


# Check missing edge weights
missing_weights = edge_df["weight"].isna().sum()

if missing_weights != 0:
    raise ValueError(
        f"Graph contains {missing_weights} missing edge weights."
    )


print("Edge validation passed.")


# ============================================================
# CREATE EDGE INDEX
# ============================================================

print("\nCreating edge_index tensor...")

edge_index_numpy = edge_df[
    [
        "source",
        "target"
    ]
].to_numpy(
    dtype=np.int64
)

edge_index = torch.tensor(
    edge_index_numpy.T,
    dtype=torch.long
)

print(
    f"edge_index shape: {edge_index.shape}"
)


# ============================================================
# CREATE EDGE WEIGHT TENSOR
# ============================================================

print("\nCreating edge weight tensor...")

edge_weight = torch.tensor(
    edge_df["weight"].to_numpy(
        dtype=np.float32
    ),
    dtype=torch.float32
)

print(
    f"edge_weight shape: {edge_weight.shape}"
)


# ============================================================
# CREATE TRAIN / VALIDATION / TEST MASKS
# ============================================================

print("\n[5/8] Creating train/validation/test masks...")

num_nodes = len(node_df)

generator = torch.Generator()

generator.manual_seed(
    RANDOM_SEED
)

random_indices = torch.randperm(
    num_nodes,
    generator=generator
)

train_size = int(
    TRAIN_RATIO * num_nodes
)

val_size = int(
    VAL_RATIO * num_nodes
)

train_indices = random_indices[
    :train_size
]

val_indices = random_indices[
    train_size:
    train_size + val_size
]

test_indices = random_indices[
    train_size + val_size:
]


# Create boolean masks

train_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)

val_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)

test_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)


train_mask[
    train_indices
] = True

val_mask[
    val_indices
] = True

test_mask[
    test_indices
] = True


print(
    f"Training nodes:   {train_mask.sum().item():,}"
)

print(
    f"Validation nodes: {val_mask.sum().item():,}"
)

print(
    f"Test nodes:       {test_mask.sum().item():,}"
)


# ============================================================
# CHECK MASK OVERLAP
# ============================================================

if torch.any(
    train_mask & val_mask
):
    raise ValueError(
        "Train and validation masks overlap."
    )

if torch.any(
    train_mask & test_mask
):
    raise ValueError(
        "Train and test masks overlap."
    )

if torch.any(
    val_mask & test_mask
):
    raise ValueError(
        "Validation and test masks overlap."
    )


if (
    train_mask.sum()
    + val_mask.sum()
    + test_mask.sum()
) != num_nodes:
    raise ValueError(
        "Train/validation/test masks do not cover all nodes."
    )

print("Mask validation passed.")


# ============================================================
# CREATE PYTORCH GEOMETRIC DATA OBJECT
# ============================================================

print("\n[6/8] Creating PyTorch Geometric Data object...")

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_weight,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)


# ============================================================
# PRINT DATA OBJECT
# ============================================================

print("\nPyTorch Geometric Data object:")

print(data)

print("\nData properties:")

print(
    f"Number of nodes:        {data.num_nodes:,}"
)

print(
    f"Number of edges:        {data.num_edges:,}"
)

print(
    f"Number of features:     {data.num_node_features}"
)

print(
    f"Number of edge weights: {data.edge_attr.shape[0]:,}"
)

print(
    f"Training nodes:         {data.train_mask.sum().item():,}"
)

print(
    f"Validation nodes:       {data.val_mask.sum().item():,}"
)

print(
    f"Test nodes:             {data.test_mask.sum().item():,}"
)


# ============================================================
# GRAPH VALIDATION
# ============================================================

print("\n[7/8] Validating PyTorch Geometric graph...")

if data.x.shape[0] != data.num_nodes:
    raise ValueError(
        "Number of node feature rows does not match nodes."
    )

if data.y.shape[0] != data.num_nodes:
    raise ValueError(
        "Number of labels does not match nodes."
    )

if data.edge_index.shape[1] != data.edge_attr.shape[0]:
    raise ValueError(
        "Number of edges does not match edge weights."
    )

if data.edge_index.min() < 0:
    raise ValueError(
        "Negative node index exists."
    )

if data.edge_index.max() >= data.num_nodes:
    raise ValueError(
        "Edge index exceeds node count."
    )

print("PyTorch Geometric validation passed.")


# ============================================================
# SAVE GNN DATA
# ============================================================

print("\nSaving GNN-ready dataset...")

torch.save(
    data,
    OUTPUT_PATH
)

print(
    f"Saved to:\n{OUTPUT_PATH}"
)


# ============================================================
# GRAPH VISUALIZATION
# ============================================================

print("\n[8/8] Creating graph visualization sample...")

# Use the first GRAPH_SAMPLE_NODES nodes.
#
# We intentionally visualize only a small subgraph.
# Plotting all 42,372 nodes would be extremely cluttered.

sample_nodes = set(
    range(
        min(
            GRAPH_SAMPLE_NODES,
            num_nodes
        )
    )
)

sample_edges = edge_df[
    edge_df["source"].isin(sample_nodes)
    &
    edge_df["target"].isin(sample_nodes)
].copy()

print(
    f"Visualization nodes: "
    f"{len(sample_nodes):,}"
)

print(
    f"Visualization edges: "
    f"{len(sample_edges):,}"
)


# Create NetworkX graph

G = nx.Graph()

G.add_nodes_from(
    sample_nodes
)

for row in sample_edges.itertuples(
    index=False
):
    G.add_edge(
        int(row.source),
        int(row.target),
        weight=float(row.weight)
    )


# Plot graph

plt.figure(
    figsize=(14, 10)
)

if len(G.nodes) > 0:

    positions = nx.spring_layout(
        G,
        seed=RANDOM_SEED,
        k=0.15,
        iterations=50
    )

    nx.draw_networkx_nodes(
        G,
        positions,
        node_size=25,
        alpha=0.8
    )

    nx.draw_networkx_edges(
        G,
        positions,
        width=0.5,
        alpha=0.25
    )

    plt.title(
        "GridGuard AI - Household Similarity Graph Sample"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        GRAPH_IMAGE_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graph visualization saved to:\n"
        f"{GRAPH_IMAGE_PATH}"
    )

else:

    plt.close()

    print(
        "Graph visualization skipped because "
        "the sample graph is empty."
    )


# ============================================================
# GRAPH STATISTICS
# ============================================================

graph_statistics = pd.DataFrame(
    {
        "metric": [
            "num_nodes",
            "num_edges",
            "num_node_features",
            "num_training_nodes",
            "num_validation_nodes",
            "num_test_nodes",
            "self_loops",
            "duplicate_edges",
            "average_degree",
            "average_edge_weight",
            "min_edge_weight",
            "max_edge_weight",
            "train_ratio",
            "validation_ratio",
            "test_ratio"
        ],
        "value": [
            data.num_nodes,
            data.num_edges,
            data.num_node_features,
            int(data.train_mask.sum().item()),
            int(data.val_mask.sum().item()),
            int(data.test_mask.sum().item()),
            int(self_loops),
            int(duplicate_edges),
            data.num_edges / data.num_nodes,
            float(edge_weight.mean()),
            float(edge_weight.min()),
            float(edge_weight.max()),
            TRAIN_RATIO,
            VAL_RATIO,
            TEST_RATIO
        ]
    }
)

graph_statistics.to_csv(
    GRAPH_STATS_PATH,
    index=False
)

print(
    f"\nGNN graph summary saved to:\n"
    f"{GRAPH_STATS_PATH}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 8 COMPLETE")
print("=" * 70)

print("\nGNN-ready graph:")

print(
    f"Nodes:             {data.num_nodes:,}"
)

print(
    f"Edges:             {data.num_edges:,}"
)

print(
    f"Node features:     {data.num_node_features}"
)

print(
    f"Train nodes:       {data.train_mask.sum().item():,}"
)

print(
    f"Validation nodes:  {data.val_mask.sum().item():,}"
)

print(
    f"Test nodes:        {data.test_mask.sum().item():,}"
)

print(
    f"Self-loops:        {self_loops}"
)

print(
    f"Duplicate edges:   {duplicate_edges}"
)

print("\nGenerated files:")

print(f"1. {OUTPUT_PATH}")
print(f"2. {GRAPH_IMAGE_PATH}")
print(f"3. {GRAPH_STATS_PATH}")

print("\nReady for Phase 9 - GraphSAGE model training.")