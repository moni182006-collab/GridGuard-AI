from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from gnn_model import GraphSAGE


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

DATA_CONFIG = CONFIG["data"]
GNN_CONFIG = CONFIG["gnn"]

DATA_DIR = BASE_DIR / DATA_CONFIG["processed_path"]
MODEL_DIR = BASE_DIR / "models"

RANDOM_STATE = CONFIG["project"]["random_state"]


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(
    RANDOM_STATE
)

np.random.seed(
    RANDOM_STATE
)


# ============================================================
# LOAD GRAPH DATA
# ============================================================

print("=" * 70)
print("GRIDGUARD AI — GRAPHSAGE TRAINING")
print("=" * 70)

try:

    data = torch.load(
        DATA_DIR / "gnn_data.pt",
        map_location="cpu",
        weights_only=False
    )

except TypeError:

    data = torch.load(
        DATA_DIR / "gnn_data.pt",
        map_location="cpu"
    )


# ============================================================
# DYNAMIC DIMENSIONS
# ============================================================

input_dim = data.num_node_features

num_classes = int(
    data.y.unique().numel()
)

# Feature-based architecture sizing.
#
# The architecture adapts to the dimensionality
# of the actual dataset.

if GNN_CONFIG[
    "hidden_dim"
]["method"] == "feature_based":

    hidden_dim = max(
        input_dim * 2,
        16
    )

else:

    raise ValueError(
        "Unsupported hidden dimension method."
    )


if GNN_CONFIG[
    "embedding_dim"
]["method"] == "feature_based":

    embedding_dim = max(
        input_dim,
        8
    )

else:

    raise ValueError(
        "Unsupported embedding dimension method."
    )


dropout = GNN_CONFIG[
    "dropout"
]

learning_rate = GNN_CONFIG[
    "learning_rate"
]

weight_decay = GNN_CONFIG[
    "weight_decay"
]

max_epochs = GNN_CONFIG[
    "max_epochs"
]

patience = GNN_CONFIG[
    "early_stopping_patience"
]


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weight_config = GNN_CONFIG[
    "class_weights"
]

class_counts = torch.bincount(
    data.y[
        data.train_mask
    ],
    minlength=num_classes
).float()

if class_weight_config[
    "method"
] == "inverse_frequency":

    class_weights = (
        class_counts.sum()
        /
        (
            num_classes
            *
            class_counts
        )
    )

else:

    raise ValueError(
        "Unsupported class-weight method."
    )


# Apply configurable suspicious-class multiplier
positive_class = int(
    data.y.max().item()
)

multiplier = class_weight_config[
    "suspicious_multiplier"
]

class_weights[
    positive_class
] *= multiplier


# ============================================================
# MODEL
# ============================================================

model = GraphSAGE(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    embedding_dim=embedding_dim,
    num_classes=num_classes,
    dropout=dropout,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# TRAINING
# ============================================================

best_validation_loss = float(
    "inf"
)

best_state = None

epochs_without_improvement = 0

history = []


for epoch in range(
    1,
    max_epochs + 1
):

    model.train()

    optimizer.zero_grad()

    logits = model(
        data.x,
        data.edge_index
    )

    train_loss = criterion(
        logits[
            data.train_mask
        ],
        data.y[
            data.train_mask
        ]
    )

    train_loss.backward()

    optimizer.step()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        validation_logits = model(
            data.x,
            data.edge_index
        )

        validation_loss = criterion(
            validation_logits[
                data.val_mask
            ],
            data.y[
                data.val_mask
            ]
        )

    validation_loss_value = (
        validation_loss.item()
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss.item(),
        "validation_loss":
            validation_loss_value,
    })


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        validation_loss_value
        <
        best_validation_loss
    ):

        best_validation_loss = (
            validation_loss_value
        )

        best_state = {
            key: value.detach().clone()
            for key, value
            in model.state_dict().items()
        }

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1


    if epoch % max(
        1,
        max_epochs // 10
    ) == 0:

        print(
            f"Epoch {epoch:04d} | "
            f"Train Loss: "
            f"{train_loss.item():.4f} | "
            f"Val Loss: "
            f"{validation_loss_value:.4f}"
        )


    if (
        epochs_without_improvement
        >= patience
    ):

        print(
            f"\nEarly stopping at epoch "
            f"{epoch}."
        )

        break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_state is None:

    raise RuntimeError(
        "No valid model state was saved."
    )

model.load_state_dict(
    best_state
)

model.eval()


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

with torch.no_grad():

    logits = model(
        data.x,
        data.edge_index
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )[:, positive_class]

    predictions = (
        logits.argmax(dim=1)
    )


test_y = data.y[
    data.test_mask
].cpu().numpy()

test_predictions = predictions[
    data.test_mask
].cpu().numpy()

test_probabilities = probabilities[
    data.test_mask
].cpu().numpy()


metrics = {
    "accuracy": accuracy_score(
        test_y,
        test_predictions
    ),
    "precision": precision_score(
        test_y,
        test_predictions,
        zero_division=0
    ),
    "recall": recall_score(
        test_y,
        test_predictions,
        zero_division=0
    ),
    "f1": f1_score(
        test_y,
        test_predictions,
        zero_division=0
    ),
    "roc_auc": roc_auc_score(
        test_y,
        test_probabilities
    ),
}


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

model_path = (
    MODEL_DIR / "gnn_model.pt"
)

torch.save(
    model.state_dict(),
    model_path
)


# ============================================================
# SAVE MODEL CONFIGURATION
# ============================================================

model_config = {
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "embedding_dim": embedding_dim,
    "num_classes": num_classes,
    "dropout": dropout,
    "positive_class": positive_class,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "max_epochs": max_epochs,
    "early_stopping_patience": patience,
    "best_validation_loss":
        best_validation_loss,
    "best_epoch":
        len(history),
    "random_state": RANDOM_STATE,
}

with open(
    MODEL_DIR / "gnn_config.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        model_config,
        file,
        indent=4
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    DATA_DIR / "gnn_training_history.csv",
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([
    metrics
])

metrics_df.to_csv(
    DATA_DIR / "gnn_metrics.csv",
    index=False
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_y,
    test_predictions,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(
    report
).transpose().to_csv(
    DATA_DIR / "gnn_classification_report.csv"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_y,
    test_predictions
)

pd.DataFrame(
    cm
).to_csv(
    DATA_DIR / "gnn_confusion_matrix.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GRAPHSAGE TRAINING COMPLETE")
print("=" * 70)

print(
    f"Nodes: {data.num_nodes:,}"
)

print(
    f"Edges: {data.edge_index.shape[1]:,}"
)

print(
    f"Features: {input_dim:,}"
)

print(
    f"Classes: {num_classes:,}"
)

print(
    f"Hidden dimension: {hidden_dim:,}"
)

print(
    f"Embedding dimension: {embedding_dim:,}"
)

print(
    "\nTest metrics:"
)

for name, value in metrics.items():

    print(
        f"{name}: {value:.4f}"
    )

print(
    f"\nModel saved:\n{model_path}"
)

print(
    "\nConfiguration saved:\n"
    f"{MODEL_DIR / 'gnn_config.json'}"
)