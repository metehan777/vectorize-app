import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import umap
import warnings
from scipy import sparse

class EmbeddingVisualizer:
    def __init__(self):
        warnings.filterwarnings('ignore')
        
    def visualize_embeddings(self, embeddings_data, method='pca'):
        """
        Visualize embeddings using dimensionality reduction
        
        Args:
            embeddings_data: List of dictionaries with content and embedding
            method: 'pca' or 'umap' for dimensionality reduction
            
        Returns:
            Plotly figure object
        """
        if not embeddings_data:
            return go.Figure().update_layout(title="No data to visualize")
        
        # Print the first item to debug the structure
        print("Data structure sample:", embeddings_data[0].keys())
        
        # Extract embeddings and texts - adapt to your actual data structure
        # Check what keys are actually in your data
        if 'content' in embeddings_data[0]:
            # If the key is 'content' instead of 'text'
            texts = [item['content'][:50] + '...' if len(item['content']) > 50 else item['content'] 
                    for item in embeddings_data]
        elif 'url' in embeddings_data[0]:
            # If there's a URL key, use that as label
            texts = [item['url'] for item in embeddings_data]
        else:
            # Fallback to using index numbers
            texts = [f"Item {i}" for i in range(len(embeddings_data))]
        
        # Extract embeddings - adapt to your actual data structure
        if 'embedding' in embeddings_data[0]:
            embeddings = np.array([item['embedding'] for item in embeddings_data])
        elif 'vector' in embeddings_data[0]:
            embeddings = np.array([item['vector'] for item in embeddings_data])
        else:
            # Try to find any numpy array in the data
            for key in embeddings_data[0].keys():
                if isinstance(embeddings_data[0][key], (list, np.ndarray)) and len(embeddings_data[0][key]) > 10:
                    embeddings = np.array([item[key] for item in embeddings_data])
                    break
            else:
                return go.Figure().update_layout(title="Could not find embedding vectors in data")
        
        # Reduce dimensions
        if method == 'pca':
            reducer = PCA(n_components=3)
            reduced_data = reducer.fit_transform(embeddings)
        else:  # umap
            reducer = umap.UMAP(n_components=3)
            reduced_data = reducer.fit_transform(embeddings)
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'x': reduced_data[:, 0],
            'y': reduced_data[:, 1],
            'z': reduced_data[:, 2] if reduced_data.shape[1] > 2 else np.zeros(len(reduced_data)),
            'text': texts
        })
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            hover_data=['text'],
            opacity=0.7,
            title=f"Embedding Visualization using {method.upper()}"
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig 