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
        
    def visualize_embeddings(self, embeddings_data, method='pca', dimensions=3):
        """
        Visualize embeddings using dimensionality reduction
        
        Args:
            embeddings_data: List of dictionaries with content and embedding
            method: 'pca' or 'umap' for dimensionality reduction
            dimensions: Number of dimensions for the visualization
            
        Returns:
            Plotly figure object
        """
        if not embeddings_data:
            return px.scatter(title="No data to visualize")
        
        # Print the first item to debug the structure
        print("Data structure sample:", embeddings_data[0].keys())
        
        # Extract embeddings, URLs, titles, and content
        embeddings = np.array([item['embedding'] for item in embeddings_data])
        urls = [item['url'] for item in embeddings_data]
        titles = [item['title'] for item in embeddings_data]
        texts = [item['content'][:200] + '...' if len(item['content']) > 200 else item['content'] for item in embeddings_data]
        
        # Reduce dimensions
        if method == 'pca':
            reducer = PCA(n_components=dimensions)
        else:  # umap
            reducer = umap.UMAP(n_components=dimensions)
        
        reduced_data = reducer.fit_transform(embeddings)
        
        # Create dataframe for plotting
        df = pd.DataFrame(reduced_data, columns=[f"Dimension {i+1}" for i in range(dimensions)])
        df['url'] = urls
        df['title'] = titles
        df['content'] = texts
        
        hover_data = {
            'url': True,
            'title': True,
            'content': True
        }
        
        if dimensions == 3:
            fig = px.scatter_3d(df, x='Dimension 1', y='Dimension 2', z='Dimension 3',
                                hover_data=hover_data,
                                title=f"{method.upper()} 3D Visualization")
            fig.update_layout(scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            ))
        else:
            fig = px.scatter(df, x='Dimension 1', y='Dimension 2',
                             hover_data=hover_data,
                             title=f"{method.upper()} 2D Visualization")
            fig.update_layout(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2'
            )
        
        return fig 