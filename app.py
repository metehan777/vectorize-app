import os
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv
from crawler import WebCrawler
from embeddings import GoogleCloudEmbeddings, EmbeddingProcessor
from visualizer import EmbeddingVisualizer
from flask_cors import CORS

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Global variables to store data
crawled_data = []
processed_data = []
visualizations = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crawl', methods=['POST'])
def crawl():
    global crawled_data
    
    try:
        # Get form data
        url = request.form.get('url')
        if not url:
            return jsonify({
                'status': 'error',
                'message': 'URL is required'
            }), 400
            
        try:
            max_pages = int(request.form.get('max_pages', 20))
        except ValueError:
            max_pages = 20
            
        same_domain = request.form.get('same_domain') == 'on'
        
        # Initialize and run crawler
        crawler = WebCrawler(url, max_pages=max_pages, same_domain_only=same_domain)
        crawled_data = crawler.crawl()
        
        if not crawled_data:
            return jsonify({
                'status': 'error',
                'message': 'Failed to crawl any pages. Please check the URL and try again.'
            }), 400
        
        return jsonify({
            'status': 'success',
            'message': f'Crawled {len(crawled_data)} pages',
            'pages': [{'url': page['url'], 'title': page['title']} for page in crawled_data]
        })
    except Exception as e:
        app.logger.error(f"Error in /crawl endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500

@app.route('/vectorize', methods=['POST'])
def vectorize():
    global crawled_data, processed_data
    
    if not crawled_data:
        return jsonify({'status': 'error', 'message': 'No crawled data available'})
    
    try:
        # Initialize embedding client with default project ID from embeddings.py
        embedding_client = GoogleCloudEmbeddings()
        processor = EmbeddingProcessor(embedding_client)
        
        # Process the crawled data
        processed_data = processor.process_pages(crawled_data)
        
        return jsonify({
            'status': 'success',
            'message': f'Vectorized {len(processed_data)} pages'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/visualize', methods=['GET'])
def visualize():
    global processed_data, visualizations
    
    if not processed_data:
        return redirect(url_for('index'))
    
    # Generate visualizations
    visualizer = EmbeddingVisualizer(processed_data)
    visualizations = visualizer.generate_visualizations()
    
    # Convert plots to JSON
    plots_json = {
        'pca_3d': visualizations['pca_3d'].to_json(),
        'pca_2d': visualizations['pca_2d'].to_json(),
        'umap_3d': visualizations['umap_3d'].to_json(),
        'umap_2d': visualizations['umap_2d'].to_json()
    }
    
    return render_template('results.html', plots=plots_json)

@app.route('/export', methods=['GET'])
def export_data():
    global processed_data
    
    if not processed_data:
        return jsonify({'status': 'error', 'message': 'No processed data available'})
    
    # Create a simplified version for export (without the large embedding vectors)
    export_data = []
    for page in processed_data:
        export_data.append({
            'url': page['url'],
            'title': page['title'],
            'content_preview': page['content'][:200] + '...' if len(page['content']) > 200 else page['content']
        })
    
    return jsonify({
        'status': 'success',
        'data': export_data
    })

if __name__ == '__main__':
    app.run(debug=True) 