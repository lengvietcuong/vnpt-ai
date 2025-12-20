"""Document scraper for Vietnamese Government documents using Selenium."""

import os
import re
import time
import json
import hashlib
import logging
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse, unquote
from typing import List, Dict, Optional, Set

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    ElementClickInterceptedException,
)
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from tqdm import tqdm

import config
from converter import DocumentConverter


# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DocumentScraper:
    """Scraper for downloading and converting government documents."""

    def __init__(self):
        """Initialize the scraper."""
        self.driver: Optional[webdriver.Chrome] = None
        self.converter = DocumentConverter()
        self.downloaded_files: List[Dict] = []
        self.failed_downloads: List[Dict] = []
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.USER_AGENT,
            'Accept': 'application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,*/*',
        })

    def _init_driver(self):
        """Initialize Selenium WebDriver."""
        logger.info("Initializing Chrome WebDriver...")
        
        chrome_options = Options()
        
        if config.SELENIUM_HEADLESS:
            chrome_options.add_argument("--headless=new")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(f"--user-agent={config.USER_AGENT}")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        
        # Set download preferences
        prefs = {
            "download.default_directory": config.DOWNLOAD_DIR,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True,
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.implicitly_wait(config.SELENIUM_IMPLICIT_WAIT)
        self.driver.set_page_load_timeout(config.SELENIUM_PAGE_LOAD_TIMEOUT)
        
        logger.info("WebDriver initialized successfully")

    def _close_driver(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("WebDriver closed")

    def _wait_for_element(self, by: By, value: str, timeout: int = None) -> Optional:
        """Wait for an element to be present."""
        timeout = timeout or config.SELENIUM_TIMEOUT
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            return None

    def _wait_for_clickable(self, by: By, value: str, timeout: int = None) -> Optional:
        """Wait for an element to be clickable."""
        timeout = timeout or config.SELENIUM_TIMEOUT
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by, value))
            )
            return element
        except TimeoutException:
            return None

    def _safe_click(self, element, retries: int = 3) -> bool:
        """Safely click an element with retries."""
        for attempt in range(retries):
            try:
                self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                time.sleep(0.5)
                element.click()
                return True
            except (StaleElementReferenceException, ElementClickInterceptedException) as e:
                if attempt == retries - 1:
                    logger.warning(f"Failed to click element after {retries} attempts: {e}")
                    return False
                time.sleep(1)
        return False

    def _extract_typegroupid(self, url: str) -> str:
        """Extract typegroupid from URL."""
        match = re.search(r'typegroupid=(\d+)', url)
        return match.group(1) if match else "unknown"

    def _get_tabs(self) -> List[Dict]:
        """Find and return all tab elements on the page."""
        tabs = []
        
        # Try different tab selectors
        for selector in config.TAB_SELECTORS:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    for elem in elements:
                        tab_text = elem.text.strip()
                        if tab_text and len(tab_text) > 0:
                            tabs.append({
                                'element': elem,
                                'name': tab_text,
                                'selector': selector
                            })
                    if tabs:
                        logger.info(f"Found {len(tabs)} tabs using selector: {selector}")
                        break
            except Exception as e:
                continue
        
        # Also try to find tabs by looking for common Vietnamese patterns
        if not tabs:
            try:
                # Look for links that might be tabs
                all_links = self.driver.find_elements(By.TAG_NAME, "a")
                for link in all_links:
                    href = link.get_attribute("href") or ""
                    text = link.text.strip()
                    # Check if it's likely a tab (has text, might have onclick or specific class)
                    classes = link.get_attribute("class") or ""
                    if text and ("tab" in classes.lower() or "nav" in classes.lower()):
                        tabs.append({
                            'element': link,
                            'name': text,
                            'selector': 'a'
                        })
            except Exception as e:
                logger.warning(f"Error finding tabs: {e}")
        
        return tabs

    def _get_document_links(self) -> List[Dict]:
        """Extract document download links from the current page."""
        documents = []
        
        try:
            # Get page source and parse with BeautifulSoup
            soup = BeautifulSoup(self.driver.page_source, 'lxml')
            
            # Look specifically for bl-doc-file divs (Vietnamese government site structure)
            doc_divs = soup.find_all('div', class_='bl-doc-file')
            logger.info(f"Found {len(doc_divs)} bl-doc-file divs")
            
            for div in doc_divs:
                link = div.find('a', href=True)
                if link:
                    href = link['href']
                    text = link.get_text(strip=True)
                    
                    # Make absolute URL
                    full_url = urljoin(self.driver.current_url, href)
                    
                    if full_url not in self.visited_urls:
                        documents.append({
                            'url': full_url,
                            'text': text or 'Unknown',
                            'filename': self._extract_filename(href, text),
                        })
                        self.visited_urls.add(full_url)
            
            # Fallback: Look for direct document links if no bl-doc-file divs found
            if not documents:
                logger.info("No bl-doc-file divs found, trying fallback method")
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    text = link.get_text(strip=True)
                    
                    # Check if it's a document link
                    is_document = False
                    for ext in config.SUPPORTED_EXTENSIONS:
                        if ext in href.lower():
                            is_document = True
                            break
                    
                    if is_document:
                        full_url = urljoin(self.driver.current_url, href)
                        
                        if full_url not in self.visited_urls:
                            documents.append({
                                'url': full_url,
                                'text': text or 'Unknown',
                                'filename': self._extract_filename(href, text),
                            })
                            self.visited_urls.add(full_url)
            
            logger.info(f"Found {len(documents)} document links on current page")
            
        except Exception as e:
            logger.error(f"Error extracting document links: {e}")
        
        return documents

    def _extract_filename(self, url: str, text: str) -> str:
        """Extract or generate a filename from URL and text."""
        # Try to get filename from URL
        parsed = urlparse(url)
        path = unquote(parsed.path)
        
        if path:
            filename = os.path.basename(path)
            if filename and any(filename.lower().endswith(ext) for ext in config.SUPPORTED_EXTENSIONS):
                return filename
        
        # Generate filename from text
        safe_text = re.sub(r'[<>:"/\\|?*]', '', text)
        safe_text = re.sub(r'\s+', '_', safe_text)
        safe_text = safe_text[:config.FILENAME_MAX_LENGTH]
        
        # Add hash for uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        # Determine extension from URL
        ext = '.pdf'  # Default
        for supported_ext in config.SUPPORTED_EXTENSIONS:
            if supported_ext in url.lower():
                ext = supported_ext
                break
        
        return f"{safe_text}_{url_hash}{ext}"

    def _has_next_page(self) -> Optional:
        """Check if there's a next page and return the element."""
        try:
            # Look for JavaScript postback links for pagination (Vietnamese government site)
            # Pattern: javascript:__doPostBack('...','Page$Next') or Page$Last with forward arrow icon
            next_links = self.driver.find_elements(
                By.XPATH,
                "//a[contains(@href, 'doPostBack') and contains(@href, 'Page$')]"
            )
            
            for link in next_links:
                try:
                    href = link.get_attribute('href') or ''
                    # Look for next page indicators
                    if 'Page$Next' in href or ('Page$' in href and link.find_elements(By.CSS_SELECTOR, 'i.fa-step-forward')):
                        if link.is_displayed() and link.is_enabled():
                            logger.debug(f"Found next page link: {href[:100]}")
                            return link
                except:
                    continue
            
            # Fallback to standard selectors
            for selector in config.PAGINATION_SELECTORS:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for elem in elements:
                        if elem.is_displayed() and elem.is_enabled():
                            return elem
                except Exception:
                    continue
            
            # Look for pagination with numbers
            try:
                # Find current page and check if there's a next number
                current_page = self.driver.find_elements(By.CSS_SELECTOR, 'span.current, a.active, .pagination .active')
                if current_page:
                    # Look for the next sibling or any higher page number
                    all_page_links = self.driver.find_elements(By.XPATH, "//a[contains(@href, 'doPostBack') and contains(@href, 'Page$')]")
                    if all_page_links:
                        # Return the first one that's not the current page
                        for page_link in all_page_links:
                            if page_link.is_displayed() and page_link.is_enabled():
                                return page_link
            except Exception:
                pass
            
        except Exception as e:
            logger.debug(f"Error checking for next page: {e}")
        
        return None

    def _download_file(self, url: str, filename: str, metadata: Dict) -> Optional[str]:
        """Download a file from URL."""
        filepath = os.path.join(config.DOWNLOAD_DIR, filename)
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            logger.info(f"File already exists: {filename}")
            return filepath
        
        for attempt in range(config.MAX_DOWNLOAD_RETRIES):
            try:
                logger.info(f"Downloading: {filename} (attempt {attempt + 1})")
                
                # Get cookies from Selenium session
                cookies = {c['name']: c['value'] for c in self.driver.get_cookies()}
                
                response = self.session.get(
                    url,
                    cookies=cookies,
                    timeout=config.REQUEST_TIMEOUT,
                    stream=True
                )
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('Content-Type', '')
                
                # Write file
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=config.CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                
                # Verify file was written
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    logger.info(f"Downloaded: {filename} ({os.path.getsize(filepath)} bytes)")
                    
                    # Save metadata
                    metadata_file = filepath + '.meta.json'
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
                    return filepath
                else:
                    logger.warning(f"Downloaded file is empty or missing: {filename}")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    
            except requests.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < config.MAX_DOWNLOAD_RETRIES - 1:
                    time.sleep(config.RETRY_DELAY)
        
        logger.error(f"Failed to download after {config.MAX_DOWNLOAD_RETRIES} attempts: {url}")
        return None

    def scrape_url(self, url: str) -> List[Dict]:
        """Scrape a single URL for documents with pagination."""
        typegroupid = self._extract_typegroupid(url)
        all_documents = []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Scraping URL: {url}")
        logger.info(f"Type Group ID: {typegroupid}")
        logger.info(f"{'='*60}")
        
        try:
            self.driver.get(url)
            time.sleep(config.PAGE_LOAD_WAIT)
            
            # Process all pages with pagination
            page_num = 1
            consecutive_failures = 0
            max_consecutive_failures = 3
            
            while True:
                logger.info(f"Processing page {page_num}")
                
                # Get document links from current page
                documents = self._get_document_links()
                
                if not documents:
                    consecutive_failures += 1
                    logger.warning(f"No documents found on page {page_num} (failure {consecutive_failures}/{max_consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"No documents found for {max_consecutive_failures} consecutive pages, stopping")
                        break
                else:
                    consecutive_failures = 0  # Reset counter on success
                    
                    for doc in documents:
                        doc['typegroupid'] = typegroupid
                        doc['page_num'] = page_num
                        all_documents.append(doc)
                
                # Check for max pages limit
                if config.MAX_PAGES_PER_TAB > 0 and page_num >= config.MAX_PAGES_PER_TAB:
                    logger.info(f"Reached max pages limit ({config.MAX_PAGES_PER_TAB})")
                    break
                
                # Try to go to next page
                next_button = self._has_next_page()
                if next_button:
                    try:
                        logger.info("Found next page button, clicking...")
                        if self._safe_click(next_button):
                            time.sleep(config.PAGE_LOAD_WAIT)
                            page_num += 1
                        else:
                            logger.info("Could not click next page button")
                            break
                    except Exception as e:
                        logger.warning(f"Error navigating to next page: {e}")
                        break
                else:
                    logger.info("No more pages found")
                    break
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}", exc_info=True)
        
        logger.info(f"Found {len(all_documents)} documents from {url} across {page_num} pages")
        return all_documents

    def download_documents(self, documents: List[Dict]) -> List[str]:
        """Download all documents."""
        downloaded_files = []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading {len(documents)} documents")
        logger.info(f"{'='*60}\n")
        
        for doc in tqdm(documents, desc="Downloading"):
            metadata = {
                'source_url': doc['url'],
                'typegroupid': doc.get('typegroupid', 'unknown'),
                'page_num': doc.get('page_num', 1),
                'original_text': doc.get('text', ''),
                'download_time': datetime.now().isoformat(),
            }
            
            filepath = self._download_file(doc['url'], doc['filename'], metadata)
            
            if filepath:
                self.downloaded_files.append({
                    'filepath': filepath,
                    'metadata': metadata,
                })
                downloaded_files.append(filepath)
                logger.info(f"✓ Downloaded: {doc['filename']}")
            else:
                self.failed_downloads.append(doc)
                logger.error(f"✗ Failed: {doc['filename']}")
            
            time.sleep(config.DOWNLOAD_DELAY)
        
        logger.info(f"\nDownload complete: {len(downloaded_files)}/{len(documents)} successful")
        return downloaded_files

    def convert_documents(self, files: List[str] = None) -> int:
        """Convert downloaded documents to markdown."""
        if files is None:
            # Get all files from downloads directory
            files = []
            for ext in config.SUPPORTED_EXTENSIONS:
                files.extend(Path(config.DOWNLOAD_DIR).glob(f"*{ext}"))
            files = [str(f) for f in files]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Converting {len(files)} documents to markdown")
        logger.info(f"{'='*60}\n")
        
        converted_count = 0
        
        for filepath in tqdm(files, desc="Converting"):
            try:
                # Load metadata if exists
                metadata_file = filepath + '.meta.json'
                metadata = {}
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                
                # Determine output path
                filename = Path(filepath).stem + '.md'
                output_path = os.path.join(config.OUTPUT_DIR, filename)
                
                # Convert
                success = self.converter.convert_and_save(filepath, output_path, metadata)
                
                if success:
                    converted_count += 1
                    
            except Exception as e:
                logger.error(f"Error converting {filepath}: {e}")
        
        return converted_count

    def run(self, download_only: bool = False, convert_only: bool = False):
        """Run the complete scraping workflow."""
        start_time = datetime.now()
        
        logger.info(f"\n{'='*60}")
        logger.info("Document Scraper Starting")
        logger.info(f"Time: {start_time.isoformat()}")
        logger.info(f"{'='*60}\n")
        
        if convert_only:
            # Only convert existing files
            converted = self.convert_documents()
            logger.info(f"Converted {converted} documents")
            return
        
        try:
            self._init_driver()
            
            # Scrape all URLs
            all_documents = []
            for url in config.TARGET_URLS:
                documents = self.scrape_url(url)
                all_documents.extend(documents)
            
            logger.info(f"\nTotal documents found: {len(all_documents)}")
            
            # Download documents
            if all_documents:
                downloaded = self.download_documents(all_documents)
                logger.info(f"Successfully downloaded: {len(downloaded)}")
                logger.info(f"Failed downloads: {len(self.failed_downloads)}")
                
                # Convert if not download_only
                if not download_only and downloaded:
                    converted = self.convert_documents(downloaded)
                    logger.info(f"Converted to markdown: {converted}")
            
        finally:
            self._close_driver()
        
        # Print summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"\n{'='*60}")
        logger.info("Scraping Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Duration: {duration}")
        logger.info(f"Documents found: {len(all_documents) if 'all_documents' in locals() else 0}")
        logger.info(f"Downloads successful: {len(self.downloaded_files)}")
        logger.info(f"Downloads failed: {len(self.failed_downloads)}")
        logger.info(f"Output directory: {config.OUTPUT_DIR}")
        
        # Save failed downloads to file
        if self.failed_downloads:
            failed_file = os.path.join(config.LOG_DIR, 'failed_downloads.json')
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(self.failed_downloads, f, ensure_ascii=False, indent=2)
            logger.info(f"Failed downloads saved to: {failed_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape government documents and convert to markdown"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download documents, don't convert"
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert existing downloaded documents"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run browser in headless mode (default: True)"
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser with visible window"
    )
    
    args = parser.parse_args()
    
    # Update config based on args
    if args.no_headless:
        config.SELENIUM_HEADLESS = False
    
    scraper = DocumentScraper()
    scraper.run(download_only=args.download_only, convert_only=args.convert_only)


if __name__ == "__main__":
    main()
