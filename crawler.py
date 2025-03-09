import requests
from bs4 import BeautifulSoup
import re
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
    
    def crawl(self, max_pages=5):
        self._crawl_recursive(self.base_url, max_pages)
        return self.pages_data
        
    def _crawl_recursive(self, url, max_pages):
        if len(self.pages_data) >= max_pages or url in self.visited_urls:
            return
            
        try:
            self.visited_urls.add(url)
            print(f"Crawling: {url}")
            
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title and content
            title = soup.title.string if soup.title else "No Title"
            
            # Get main content (this is a simple approach, might need refinement)
            content = ""
            for paragraph in soup.find_all('p'):
                content += paragraph.get_text() + "\n"
                
            # Store the data
            self.pages_data.append({
                'url': url,
                'title': title,
                'content': content
            })
            
            # If we've reached our limit, stop
            if len(self.pages_data) >= max_pages:  # Limit to max_pages
                return
                
            # Find links and crawl them
            links = self.extract_links(soup, url)
            for link in links:
                if len(self.pages_data) >= max_pages:
                    break
                self._crawl_recursive(link, max_pages)
                    
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            
    def _extract_links(self, soup, current_url):
        links = []
        base_domain = urlparse(self.base_url).netloc
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = urljoin(current_url, href)
            
            # Only include links to the same domain
            if urlparse(absolute_url).netloc == base_domain:
                links.append(absolute_url)
                
        return links 