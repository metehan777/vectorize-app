import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json

class WebCrawler:
    def __init__(self, base_url, max_pages=50, same_domain_only=True):
        self.base_url = base_url
        self.max_pages = max_pages
        self.same_domain_only = same_domain_only
        self.visited_urls = set()
        self.to_visit = [base_url]
        self.domain = urlparse(base_url).netloc
        self.pages_data = []
        
    def is_valid_url(self, url):
        """Check if URL should be crawled."""
        parsed = urlparse(url)
        
        # Skip non-http(s) URLs
        if parsed.scheme not in ('http', 'https'):
            return False
            
        # Skip if we want to stay on the same domain but URL is external
        if self.same_domain_only and parsed.netloc != self.domain:
            return False
            
        return True
    
    def extract_text(self, soup):
        """Extract meaningful text from the page."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Remove blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_links(self, soup, current_url):
        """Extract links from the page."""
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Convert relative URLs to absolute
            absolute_url = urljoin(current_url, href)
            # Remove fragments
            absolute_url = absolute_url.split('#')[0]
            if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                links.append(absolute_url)
        return links
    
    def crawl(self, progress_callback=None):
        """Start crawling the website."""
        try:
            # Validate the base URL first
            if not self.base_url.startswith(('http://', 'https://')):
                print(f"Invalid URL format: {self.base_url}. URL must start with http:// or https://")
                return []
                
            # Try to make a test request to the base URL
            try:
                test_response = requests.get(self.base_url, timeout=10)
                test_response.raise_for_status()  # Raise an exception for 4XX/5XX responses
            except requests.exceptions.RequestException as e:
                print(f"Error accessing base URL {self.base_url}: {e}")
                return []
                
            # Continue with the regular crawling process
            while self.to_visit and len(self.visited_urls) < self.max_pages:
                # Get the next URL to visit
                current_url = self.to_visit.pop(0)
                
                # Skip if already visited
                if current_url in self.visited_urls:
                    continue
                    
                print(f"Crawling: {current_url}")
                if progress_callback:
                    progress_callback(f"Crawling: {current_url}")
                
                try:
                    # Add a small delay to be respectful
                    time.sleep(1)
                    
                    # Fetch the page
                    response = requests.get(current_url, timeout=10)
                    
                    # Skip if not HTML
                    if 'text/html' not in response.headers.get('Content-Type', ''):
                        continue
                        
                    # Mark as visited
                    self.visited_urls.add(current_url)
                    
                    # Parse the HTML
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract title
                    title = soup.title.string if soup.title else "No Title"
                    
                    # Extract text content
                    text = self.extract_text(soup)
                    
                    # Store the page data
                    self.pages_data.append({
                        'url': current_url,
                        'title': title,
                        'content': text
                    })
                    
                    # Extract links and add to queue
                    links = self.extract_links(soup, current_url)
                    for link in links:
                        if link not in self.visited_urls and link not in self.to_visit:
                            self.to_visit.append(link)
                            
                except Exception as e:
                    print(f"Error crawling {current_url}: {e}")
                    
            print(f"Crawling complete. Visited {len(self.visited_urls)} pages.")
            if progress_callback:
                progress_callback(f"Crawling complete. Visited {len(self.visited_urls)} pages.")
            return self.pages_data
            
        except Exception as e:
            print(f"Unexpected error during crawling: {e}")
            if progress_callback:
                progress_callback(f"Error: {e}")
            return self.pages_data  # Return whatever we've collected so far 