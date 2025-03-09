import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import umap
import warnings
from scipy import sparse

class EmbeddingVisualizer:
    def __init__(self, pages_data):
        self.pages_data = pages_data
        self.embeddings = np.array([page['embedding'] for page in pages_data])
        print(f"Visualizing {len(pages_data)} pages with embeddings of shape {self.embeddings.shape}")
        
    def reduce_dimensions_pca(self, n_components=3):
        """Reduce dimensions using PCA."""
        # Ensure n_components doesn't exceed the number of samples
        n_samples = self.embeddings.shape[0]
        n_components = min(n_components, n_samples - 1) if n_samples > 1 else 1
        
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(self.embeddings)
        
        # Add a small amount of random noise to prevent exact overlaps
        if len(self.pages_data) > 1:
            noise_scale = 0.01 * np.std(reduced_data, axis=0)
            reduced_data += np.random.normal(0, noise_scale, reduced_data.shape)
            
        # If we only have one sample, duplicate it with slight variation for visualization
        if n_samples == 1:
            reduced_data = np.vstack([reduced_data, reduced_data + 0.1])
            
        return reduced_data
    
    def reduce_dimensions_umap(self, n_components=3):
        """Reduce dimensions using UMAP."""
        # Check if we have enough data points for UMAP
        n_samples = self.embeddings.shape[0]
        
        # If we have too few samples, fall back to PCA
        if n_samples < 4:  # UMAP needs at least 4 samples to work properly
            warnings.warn(f"Too few samples ({n_samples}) for UMAP. Falling back to PCA.")
            return self.reduce_dimensions_pca(n_components)
        
        try:
            # Adjust UMAP parameters based on sample size
            n_neighbors = min(n_samples - 1, 15)  # Default is 15, but can't exceed n_samples - 1
            min_dist = 0.1
            
            # For small datasets, use PCA first to reduce dimensions
            if n_samples < 10:
                # Use PCA to reduce to a smaller dimension first
                pca_components = min(50, n_samples - 1)
                pca_data = PCA(n_components=pca_components).fit_transform(self.embeddings)
                
                # Configure UMAP for small dataset
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric='euclidean',
                    random_state=42
                )
                
                # Convert to dense array if sparse
                if sparse.issparse(pca_data):
                    pca_data = pca_data.toarray()
                    
                return reducer.fit_transform(pca_data)
            else:
                # Use standard UMAP for larger datasets
                reducer = umap.UMAP(
                    n_components=n_components,
                    random_state=42
                )
                return reducer.fit_transform(self.embeddings)
                
        except Exception as e:
            warnings.warn(f"UMAP reduction failed: {str(e)}. Falling back to PCA.")
            # Fall back to PCA if UMAP fails
            return self.reduce_dimensions_pca(n_components)
    
    def create_3d_scatter(self, reduced_data, method="PCA"):
        """Create a 3D scatter plot of the reduced embeddings."""
        # Handle the case where we duplicated a single point for visualization
        n_actual_samples = len(self.pages_data)
        reduced_data = reduced_data[:n_actual_samples]
        
        df = pd.DataFrame({
            'x': reduced_data[:, 0],
            'y': reduced_data[:, 1],
            'z': reduced_data[:, 2] if reduced_data.shape[1] > 2 else np.zeros(reduced_data.shape[0]),
            'title': [page['title'] for page in self.pages_data],
            'url': [page['url'] for page in self.pages_data]
        })
        
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            hover_data=['title', 'url'],
            title=f"Website Content Embeddings ({method})"
        )
        
        fig.update_traces(
            marker=dict(size=5),
            hovertemplate="<b>%{customdata[0]}</b><br>URL: %{customdata[1]}<extra></extra>"
        )
        
        return fig
    
    def create_2d_scatter(self, reduced_data, method="PCA"):
        """Create a 2D scatter plot of the reduced embeddings."""
        # Handle the case where we duplicated a single point for visualization
        n_actual_samples = len(self.pages_data)
        reduced_data = reduced_data[:n_actual_samples]
        
        df = pd.DataFrame({
            'x': reduced_data[:, 0],
            'y': reduced_data[:, 1],
            'title': [page['title'] for page in self.pages_data],
            'url': [page['url'] for page in self.pages_data]
        })
        
        fig = px.scatter(
            df, x='x', y='y',
            hover_data=['title', 'url'],
            title=f"Website Content Embeddings ({method})"
        )
        
        fig.update_traces(
            marker=dict(size=8),
            hovertemplate="<b>%{customdata[0]}</b><br>URL: %{customdata[1]}<extra></extra>"
        )
        
        return fig
    
    def generate_visualizations(self):
        """Generate all visualizations."""
        # PCA visualizations
        pca_3d = self.reduce_dimensions_pca(n_components=3)
        pca_2d = self.reduce_dimensions_pca(n_components=2)
        
        # UMAP visualizations (with fallback to PCA if needed)
        try:
            umap_3d = self.reduce_dimensions_umap(n_components=3)
            umap_2d = self.reduce_dimensions_umap(n_components=2)
        except Exception as e:
            print(f"Error generating UMAP visualizations: {e}")
            # Fall back to PCA if UMAP fails completely
            umap_3d = pca_3d
            umap_2d = pca_2d
        
        # Create plots
        plots = {
            'pca_3d': self.create_3d_scatter(pca_3d, "PCA"),
            'pca_2d': self.create_2d_scatter(pca_2d, "PCA"),
            'umap_3d': self.create_3d_scatter(umap_3d, "UMAP"),
            'umap_2d': self.create_2d_scatter(umap_2d, "UMAP")
        }
        
        return plots 