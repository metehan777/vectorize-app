import streamlit as st
from crawler import WebCrawler
from embeddings import GoogleCloudEmbeddings, EmbeddingProcessor
from visualizer import EmbeddingVisualizer
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Vectorize",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

if 'visualizations' not in st.session_state:
    st.session_state.visualizations = None

# App title and description
st.title("Vectorize")
st.subheader("Web content analysis and visualization using embeddings")

# Sidebar for inputs
with st.sidebar:
    st.header("Input")
    url = st.text_input("Enter a URL to analyze:")
    max_pages = st.slider("Maximum pages to crawl:", 1, 5, 3)
    process_button = st.button("Process URL")

# Main content area
if process_button and url:
    st.session_state.processed_data = None
    st.session_state.visualizations = None
    
    with st.spinner(f"Crawling website (max {max_pages} pages)..."):
        crawler = WebCrawler(base_url=url)
        content = crawler.crawl(max_pages=max_pages)
    
    with st.spinner("Generating embeddings..."):
        embedding_client = GoogleCloudEmbeddings()
        embedding_processor = EmbeddingProcessor(embedding_client=embedding_client)
        
        embeddings_data = embedding_processor.generate_embeddings(content)
        
        st.session_state.processed_data = embeddings_data
    
    if st.session_state.processed_data:
        with st.spinner("Generating visualizations..."):
            visualizer = EmbeddingVisualizer()
            fig = visualizer.visualize_embeddings(st.session_state.processed_data, method='pca')
            st.session_state.visualizations = fig
            
            st.subheader("Content Statistics")
            st.write(f"Total pages crawled: {len(content)}")
            st.write(f"Total embeddings generated: {len(embeddings_data)}")

# Display visualizations if available
if st.session_state.processed_data is not None:
    # Make sure visualizations are generated
    if st.session_state.visualizations is None:
        with st.spinner("Generating visualizations..."):
            visualizer = EmbeddingVisualizer()
            fig = visualizer.visualize_embeddings(st.session_state.processed_data, method='pca')
            st.session_state.visualizations = fig
    
    # Display the visualization
    st.subheader("Embedding Visualization")
    st.plotly_chart(st.session_state.visualizations, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Powered by Google Generative AI and Streamlit")

# Add a guide for interpreting results
st.markdown("---")
st.subheader("📊 How to Interpret These Visualizations")

with st.expander("Click here for a detailed guide on understanding these visualizations", expanded=True):
    st.markdown("""
    ### Understanding Embedding Visualizations

    These visualizations represent the semantic relationships between the content of different pages on the website you crawled. Pages with similar content appear closer together in the visualization.

    #### Visualization Types

    1. **PCA (Principal Component Analysis)**
       - A dimensionality reduction technique that preserves global structure
       - Better at showing overall patterns and outliers
       - More stable with small datasets

    2. **UMAP (Uniform Manifold Approximation and Projection)**
       - A more advanced technique that better preserves local relationships
       - Shows clusters more clearly
       - May fall back to PCA for very small datasets (fewer than 4 pages)

    #### 2D vs 3D Views

    - **3D Views**: Provide more information but can be harder to interpret at first
    - **2D Views**: Simpler but may lose some relationship details

    #### How to Explore the Visualizations

    - **Rotate**: Click and drag to rotate 3D plots
    - **Zoom**: Use the scroll wheel or pinch gesture
    - **Pan**: Right-click and drag (or two-finger drag)
    - **Hover**: Mouse over points to see page titles and URLs
    - **Reset View**: Double-click to reset the view
    - **Toolbar**: Use the toolbar in the top-right for additional options:
      - 📷 Download as PNG
      - 🔍 Zoom
      - 🔄 Reset axes
      - ⚙️ Configure plot

    #### What to Look For

    1. **Clusters**: Groups of pages that are close together likely have similar content
    2. **Outliers**: Pages far from others may have unique content
    3. **Gradients**: Smooth transitions between topics
    4. **Dimensions**: Each axis represents a mathematical combination of features from the original embeddings

    #### Practical Applications

    - **Content Organization**: Identify logical groupings for site structure
    - **Content Gaps**: Find areas where content is sparse
    - **Redundancy**: Identify pages with very similar content
    - **Navigation Improvements**: Create better internal linking between related pages
    
    #### Technical Notes
    
    - The embeddings are generated using Google's embedding model
    - Each page's content is represented as a 768-dimensional vector
    - These visualizations reduce those dimensions to 2D or 3D for visualization
    - Small random noise is added to prevent exact overlaps
    """)

# Add a section for troubleshooting
with st.expander("Troubleshooting"):
    st.markdown("""
    ### Common Issues

    1. **Points are too close or overlapping**
       - This means the content is very similar between pages
       - Try crawling more diverse pages from the website

    2. **Only seeing one point**
       - Make sure you crawled multiple pages
       - Check if the pages have enough unique content

    3. **UMAP warnings**
       - UMAP works best with larger datasets (10+ pages)
       - For small datasets, the system automatically falls back to PCA

    4. **Slow performance**
       - Reduce the number of pages crawled
       - Close other applications to free up memory
    """)

# Add a section for next steps
with st.expander("Next Steps"):
    st.markdown("""
    ### What to Do With These Insights

    1. **Content Strategy**
       - Identify content clusters and gaps
       - Plan new content to fill sparse areas

    2. **SEO Improvements**
       - Find pages with similar topics that could be consolidated
       - Identify opportunities for internal linking

    3. **User Experience**
       - Improve navigation between related pages
       - Create better content hierarchies based on semantic relationships

    4. **Advanced Analysis**
       - Export the data for further analysis in other tools
       - Combine with analytics data to identify high-performing content clusters
    """)

# Add requirements for Streamlit to requirements.txt
st.sidebar.markdown("---")
st.sidebar.info("""
**Note:** To run this Streamlit app, add the following to requirements.txt:
streamlit==1.28.0
```
""") 