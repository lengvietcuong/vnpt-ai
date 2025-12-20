"""Web scraper that extracts articles and converts them to markdown."""

import os
import re
import asyncio
import hashlib
from urllib.parse import urljoin, urlparse
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup
from main_content_extractor import MainContentExtractor
from markdownify import markdownify as md
from tqdm import tqdm

import config


class WebScraper:
    """A web scraper that crawls websites and saves articles as markdown."""

    def __init__(self, base_url: str, output_dir: str = "output", concurrency: int = 20):
        """
        Initialize the scraper.

        Args:
            base_url: The starting URL of the website to scrape
            output_dir: Directory to save markdown files
            concurrency: Number of concurrent requests
        """
        self.base_url = base_url.rstrip("/")
        self.domain = urlparse(base_url).netloc
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.visited_urls: set[str] = set()
        self.urls_to_visit: asyncio.Queue = asyncio.Queue()
        self.saved_files: list[str] = []
        self.failed_urls: list[str] = []
        self.lock = asyncio.Lock()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL should be crawled."""
        parsed = urlparse(url)

        # Must be same domain
        if parsed.netloc and parsed.netloc != self.domain:
            return False

        # Skip certain file extensions
        path_lower = parsed.path.lower()
        for ext in config.SKIP_EXTENSIONS:
            if path_lower.endswith(ext):
                return False

        # Skip certain URL patterns
        for pattern in config.SKIP_URL_PATTERNS:
            if pattern in url:
                return False

        return True

    def _normalize_url(self, url: str, current_url: str) -> Optional[str]:
        """Convert relative URL to absolute and normalize it."""
        # Handle relative URLs
        full_url = urljoin(current_url, url)

        # Parse and reconstruct to normalize
        parsed = urlparse(full_url)

        # Only keep http/https URLs
        if parsed.scheme not in ("http", "https"):
            return None

        # Remove fragments and normalize
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        # Remove trailing slash for consistency (except for root)
        if normalized != f"{parsed.scheme}://{parsed.netloc}/":
            normalized = normalized.rstrip("/")

        return normalized

    async def _fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch a page and return its HTML content."""
        for attempt in range(config.MAX_RETRIES):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)) as response:
                    if response.status != 200:
                        return None

                    # Check if it's HTML
                    content_type = response.headers.get("content-type", "")
                    if "text/html" not in content_type.lower():
                        return None

                    return await response.text()

            except Exception as e:
                if attempt == config.MAX_RETRIES - 1:
                    return None
                await asyncio.sleep(0.5)

        return None

    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> list[str]:
        """Extract all valid links from a page."""
        links = []

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            normalized = self._normalize_url(href, current_url)

            if normalized and self._is_valid_url(normalized):
                links.append(normalized)

        return list(set(links))  # Remove duplicates

    def _extract_article_content(self, html: str) -> Optional[str]:
        """Extract the main article content from HTML using MainContentExtractor."""
        try:
            # Use MainContentExtractor to get clean article HTML
            extracted_html = MainContentExtractor.extract(html)
            
            if not extracted_html or len(extracted_html.strip()) < 100:
                return None
            
            return extracted_html
        except Exception as e:
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the page title."""
        # Try h1 first
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)

        # Try title tag
        title = soup.find("title")
        if title:
            return title.get_text(strip=True)

        return "Untitled"

    def _convert_to_markdown(self, html_content: str, title: str, url: str) -> str:
        """Convert HTML content to markdown using markdownify."""
        # Convert HTML to markdown
        markdown_content = md(
            html_content,
            heading_style="ATX",
            bullets="-",
            strip=["script", "style"],
        )

        # Clean up excessive whitespace
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        markdown_content = markdown_content.strip()

        # Add metadata header
        header = f"# {title}\n\n"
        header += f"> Source: {url}\n\n"
        header += "---\n\n"

        return header + markdown_content

    def _generate_filename(self, url: str, title: str) -> str:
        """Generate a safe filename from URL and title."""
        # Clean title for filename
        safe_title = re.sub(r'[<>:"/\\|?*]', "", title)
        safe_title = re.sub(r"\s+", "_", safe_title)
        safe_title = safe_title[:50]  # Limit length

        # Add hash of URL for uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]

        return f"{safe_title}_{url_hash}.md"

    def _save_markdown(self, content: str, filename: str) -> str:
        """Save markdown content to file."""
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath

    async def _process_url(self, session: aiohttp.ClientSession, url: str, pbar: tqdm) -> list[str]:
        """Process a single URL and return new links found."""
        html = await self._fetch_page(session, url)
        if not html:
            async with self.lock:
                self.failed_urls.append(url)
            pbar.update(1)
            return []

        soup = BeautifulSoup(html, "lxml")

        # Extract links for further crawling
        new_links = self._extract_links(soup, url)

        # Try to extract and save article using MainContentExtractor
        extracted_html = self._extract_article_content(html)
        if extracted_html:
            title = self._extract_title(soup)
            markdown = self._convert_to_markdown(extracted_html, title, url)
            filename = self._generate_filename(url, title)
            filepath = self._save_markdown(markdown, filename)
            async with self.lock:
                self.saved_files.append(filepath)
            tqdm.write(f"Saved: {filename}")

        pbar.update(1)
        return new_links

    async def _worker(self, session: aiohttp.ClientSession, pbar: tqdm, max_pages: int):
        """Worker that processes URLs from the queue."""
        while True:
            try:
                url = await asyncio.wait_for(self.urls_to_visit.get(), timeout=5.0)
            except asyncio.TimeoutError:
                # No more URLs to process
                break

            # Check if we should stop
            async with self.lock:
                if max_pages > 0 and len(self.visited_urls) >= max_pages:
                    self.urls_to_visit.task_done()
                    break

                if url in self.visited_urls:
                    self.urls_to_visit.task_done()
                    continue

                self.visited_urls.add(url)

            # Process the URL
            new_links = await self._process_url(session, url, pbar)

            # Add new links to queue
            async with self.lock:
                for link in new_links:
                    if link not in self.visited_urls:
                        await self.urls_to_visit.put(link)

            self.urls_to_visit.task_done()

            # Small delay to be polite
            await asyncio.sleep(config.REQUEST_DELAY / self.concurrency)

    async def crawl_async(self, max_pages: int = 0) -> dict:
        """
        Crawl the website starting from base_url.

        Args:
            max_pages: Maximum number of pages to crawl (0 = unlimited)

        Returns:
            Dictionary with crawl statistics
        """
        print(f"\nStarting crawl of {self.base_url}")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
        print(f"Max pages: {'unlimited' if max_pages == 0 else max_pages}")
        print(f"Concurrency: {self.concurrency} workers\n")

        # Add starting URL to queue
        await self.urls_to_visit.put(self.base_url)

        # Create HTTP session with headers
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        async with aiohttp.ClientSession(
            connector=connector,
            headers=config.DEFAULT_HEADERS
        ) as session:
            with tqdm(total=max_pages if max_pages > 0 else None, desc="Crawling") as pbar:
                # Create worker tasks
                workers = [
                    asyncio.create_task(self._worker(session, pbar, max_pages))
                    for _ in range(self.concurrency)
                ]

                # Wait for all workers to complete
                await asyncio.gather(*workers)

        # Print summary
        stats = {
            "pages_visited": len(self.visited_urls),
            "articles_saved": len(self.saved_files),
            "failed_urls": len(self.failed_urls),
            "saved_files": self.saved_files,
        }

        print(f"\n{'='*50}")
        print("Crawl Complete!")
        print(f"{'='*50}")
        print(f"Pages visited: {stats['pages_visited']}")
        print(f"Articles saved: {stats['articles_saved']}")
        print(f"Failed URLs: {stats['failed_urls']}")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")

        return stats

    def crawl(self, max_pages: int = 0) -> dict:
        """Synchronous wrapper for crawl_async."""
        return asyncio.run(self.crawl_async(max_pages))


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape a website and save articles as markdown files."
    )
    parser.add_argument(
        "url",
        help="The URL of the website to scrape"
    )
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output directory for markdown files (default: output)"
    )
    parser.add_argument(
        "-m", "--max-pages",
        type=int,
        default=0,
        help="Maximum number of pages to crawl (0 = unlimited, default: 0)"
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=20,
        help="Number of concurrent requests (default: 20)"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Only scrape the single URL provided (no crawling)"
    )

    args = parser.parse_args()

    scraper = WebScraper(args.url, args.output, args.concurrency)

    if args.single:
        # For single page, use sync approach
        async def scrape_single():
            connector = aiohttp.TCPConnector(limit=1)
            async with aiohttp.ClientSession(
                connector=connector,
                headers=config.DEFAULT_HEADERS
            ) as session:
                html = await scraper._fetch_page(session, args.url)
                if not html:
                    print("Failed to fetch the page")
                    return

                soup = BeautifulSoup(html, "lxml")
                extracted_html = scraper._extract_article_content(html)
                if not extracted_html:
                    print("Failed to extract article content")
                    return

                title = scraper._extract_title(soup)
                markdown = scraper._convert_to_markdown(extracted_html, title, args.url)
                filename = scraper._generate_filename(args.url, title)
                filepath = scraper._save_markdown(markdown, filename)
                print(f"Saved: {filepath}")

        asyncio.run(scrape_single())
    else:
        scraper.crawl(max_pages=args.max_pages)


if __name__ == "__main__":
    main()
