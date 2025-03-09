import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import time
from crawler import WebCrawler
from embeddings import GoogleCloudEmbeddings, EmbeddingProcessor
from visualizer import EmbeddingVisualizer

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Website Crawler & Vectorizer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'crawled_data' not in st.session_state:
    st.session_state.crawled_data = []
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = {}
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

# Title and description
st.title("Website Crawler & Vectorizer")
st.markdown("Crawl a website, vectorize the content using Google Cloud's embedding model, and visualize the results.")

# Sidebar for navigation
st.sidebar.title("Navigation")
step = st.sidebar.radio(
    "Go to step:",
    ["1. Crawl Website", "2. Vectorize Content", "3. Visualize Embeddings"],
    index=st.session_state.current_step - 1,
    disabled=(st.session_state.current_step == 1 and len(st.session_state.crawled_data) == 0) or 
             (st.session_state.current_step <= 2 and len(st.session_state.processed_data) == 0)
)

# Update current step based on sidebar selection
if step == "1. Crawl Website":
    st.session_state.current_step = 1
elif step == "2. Vectorize Content":
    st.session_state.current_step = 2
elif step == "3. Visualize Embeddings":
    st.session_state.current_step = 3

# Step 1: Crawl Website
if st.session_state.current_step == 1:
    st.header("Step 1: Crawl a Website")
    
    with st.form("crawl_form"):
        url = st.text_input("Website URL", placeholder="https://example.com")
        col1, col2 = st.columns(2)
        with col1:
            max_pages = st.number_input("Maximum Pages to Crawl", min_value=1, max_value=100, value=20)
        with col2:
            same_domain = st.checkbox("Stay on the same domain", value=True)
        
        submitted = st.form_submit_button("Start Crawling")
        
        if submitted:
            if not url:
                st.error("Please enter a valid URL")
            else:
                try:
                    progress_placeholder = st.empty()
                    
                    # Define a callback to update progress
                    def update_progress(message):
                        progress_placeholder.text(message)
                    
                    with st.spinner("Crawling website..."):
                        # Initialize and run crawler with progress callback
                        crawler = WebCrawler(url, max_pages=max_pages, same_domain_only=same_domain)
                        result = crawler.crawl(progress_callback=update_progress)
                        
                        if not result:
                            st.error("Failed to crawl any pages. Please check the URL and try again.")
                        else:
                            st.session_state.crawled_data = result
                            progress_placeholder.empty()  # Clear the progress messages
                            st.success(f"Crawled {len(st.session_state.crawled_data)} pages")
                            
                            # Display crawled pages
                            st.subheader("Crawled Pages")
                            df = pd.DataFrame([
                                {"Title": page["title"], "URL": page["url"]} 
                                for page in st.session_state.crawled_data
                            ])
                            st.dataframe(df, use_container_width=True)
                            
                            # Enable next step
                            st.session_state.current_step = 2
                            st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Step 2: Vectorize Content
elif st.session_state.current_step == 2:
    st.header("Step 2: Vectorize Content")
    
    st.info("Generate embeddings for the crawled content using Google Cloud's embedding model.")
    
    if st.button("Generate Embeddings"):
        with st.spinner("Generating embeddings..."):
            try:
                # Initialize embedding client with default project ID
                embedding_client = GoogleCloudEmbeddings()
                processor = EmbeddingProcessor(embedding_client)
                
                # Process the crawled data
                st.session_state.processed_data = processor.process_pages(st.session_state.crawled_data)
                
                st.success(f"Vectorized {len(st.session_state.processed_data)} pages")
                st.session_state.current_step = 3
                st.rerun()
            except Exception as e:
                st.error(f"Error generating embeddings: {str(e)}")
                
                # For demo purposes, allow continuing with dummy data
                if st.button("Continue with Demo Data"):
                    # Create dummy embeddings
                    for page in st.session_state.crawled_data:
                        # Create a random embedding vector
                        page['embedding'] = np.random.rand(768)
                    
                    st.session_state.processed_data = st.session_state.crawled_data
                    st.success("Created demo embeddings for visualization")
                    st.session_state.current_step = 3
                    st.rerun()

# Step 3: Visualize Embeddings
elif st.session_state.current_step == 3:
    st.header("Step 3: Visualize Embeddings")
    
    # Generate visualizations if not already done
    if not st.session_state.visualizations:
        with st.spinner("Generating visualizations..."):
            visualizer = EmbeddingVisualizer(st.session_state.processed_data)
            st.session_state.visualizations = visualizer.generate_visualizations()
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["PCA 3D", "PCA 2D", "UMAP 3D", "UMAP 2D"])
    
    with tab1:
        st.plotly_chart(st.session_state.visualizations['pca_3d'], use_container_width=True)
    
    with tab2:
        st.plotly_chart(st.session_state.visualizations['pca_2d'], use_container_width=True)
    
    with tab3:
        st.plotly_chart(st.session_state.visualizations['umap_3d'], use_container_width=True)
    
    with tab4:
        st.plotly_chart(st.session_state.visualizations['umap_2d'], use_container_width=True)
    
    # Export data
    st.subheader("Export Data")
    
    if st.button("Export to CSV"):
        # Create a simplified version for export (without the large embedding vectors)
        export_data = []
        for page in st.session_state.processed_data:
            export_data.append({
                'url': page['url'],
                'title': page['title'],
                'content_preview': page['content'][:200] + '...' if len(page['content']) > 200 else page['content']
            })
        
        # Convert to DataFrame and generate CSV
        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="website_data.csv",
            mime="text/csv"
        )
    
    # Display data table
    st.subheader("Crawled Data")
    
    # Create a simplified DataFrame for display
    display_data = []
    for page in st.session_state.processed_data:
        display_data.append({
            'Title': page['title'],
            'URL': page['url'],
            'Content Preview': page['content'][:100] + '...' if len(page['content']) > 100 else page['content']
        })
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True)

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