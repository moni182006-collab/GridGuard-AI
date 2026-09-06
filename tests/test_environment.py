import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn
import plotly
import networkx
import streamlit
import torch
import torch_geometric
from groq import Groq

print("================================")
print("GRIDGUARD AI ENVIRONMENT CHECK")
print("================================")

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Seaborn:", seaborn.__version__)
print("Plotly:", plotly.__version__)
print("NetworkX:", networkx.__version__)
print("Streamlit:", streamlit.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch Geometric:", torch_geometric.__version__)

print("CUDA available:", torch.cuda.is_available())

print()
print("All imports successful!")
print("================================")