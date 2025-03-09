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
            embeddings_data: List of dictionaries with text and embedding
            method: 'pca' or 'umap' for dimensionality reduction
            
        Returns:
            Plotly figure object
        """
        if not embeddings_data:
            return go.Figure().update_layout(title="No data to visualize")
        
        # Extract embeddings and texts
        texts = [item['text'][:50] + '...' if len(item['text']) > 50 else item['text'] 
                for item in embeddings_data]
        embeddings = np.array([item['embedding'] for item in embeddings_data])
        
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