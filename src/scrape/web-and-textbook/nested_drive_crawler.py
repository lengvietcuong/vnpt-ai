"""3-Level nested crawler: Main page -> Sub-pages -> Final pages with Drive links."""

import os
import re
import asyncio
from urllib.parse import urljoin, urlparse, parse_qs
from typing import Optional, List, Set, Dict
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging

import config


class NestedDriveCrawler:
    """
    3-Level crawler for nested page structures leading to Drive PDFs.
    
    Level 1: Main page with links
    Level 2: Each link has more sub-links  
    Level 3: Each sub-link contains a Drive PDF
    """

    def __init__(self, main_url: str = None, url_file: str = None, output_dir: str = "downloaded_pdfs", concurrency: int = 5):
        """
        Initialize the nested crawler.

        Args:
            main_url: The main URL containing initial list of links (or None if using url_file)
            url_file: Path to file containing URLs (one per line)
            output_dir: Directory to save downloaded PDFs
            concurrency: Number of concurrent downloads
        """
        if not main_url and not url_file:
            raise ValueError("Either main_url or url_file must be provided")
        
        self.url_file = url_file
        self.main_urls = []
        
        if url_file:
            # Read URLs from file
            with open(url_file, 'r', encoding='utf-8') as f:
                self.main_urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            if not self.main_urls:
                raise ValueError(f"No valid URLs found in {url_file}")
            self.main_url = self.main_urls[0]  # Use first URL for domain detection
        else:
            self.main_url = main_url.rstrip("/")
            self.main_urls = [self.main_url]
        
        self.main_domain = urlparse(self.main_url).netloc
        self.output_dir = Path(output_dir)
        self.concurrency = concurrency
        
        # Track crawl progress
        self.level1_links: List[str] = []  # Links from main page
        self.level2_links: Dict[str, List[str]] = {}  # Links from each level 1 page
        self.level3_drive_links: Dict[str, str] = {}  # Drive links from level 2 pages
        
        self.downloaded_files: List[str] = []
        self.failed_downloads: List[dict] = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _extract_drive_file_id(self, url: str) -> Optional[str]:
        """Extract Google Drive file ID from various Drive URL formats."""
        # Pattern 1: /file/d/{FILE_ID}/
        match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
        if match:
            return match.group(1)
        
        # Pattern 2: ?id={FILE_ID} or &id={FILE_ID}
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if 'id' in params:
            return params['id'][0]
        
        return None

    def _is_drive_link(self, url: str) -> bool:
        """Check if URL is a Google Drive link."""
        return 'drive.google.com' in url.lower()

    def _get_drive_download_url(self, file_id: str) -> str:
        """Convert Drive file ID to direct download URL."""
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    def _is_same_domain(self, url: str) -> bool:
        """Check if URL is from the same domain as main URL."""
        parsed = urlparse(url)
        return parsed.netloc == self.main_domain

    async def _fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch a page and return its HTML content."""
        for attempt in range(config.MAX_RETRIES):
            try:
                async with session.get(
                    url, 
                    timeout=aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT),
                    allow_redirects=True
                ) as response:
                    if response.status != 200:
                        return None

                    content_type = response.headers.get("content-type", "")
                    if "text/html" not in content_type.lower():
                        return None

                    return await response.text()

            except Exception as e:
                if attempt == config.MAX_RETRIES - 1:
                    self.logger.error(f"Failed to fetch {url}: {str(e)}")
                    return None
                await asyncio.sleep(0.5)

        return None

    def _extract_links(self, html: str, current_url: str, same_domain_only: bool = True) -> List[str]:
        """Extract all links from HTML."""
        soup = BeautifulSoup(html, "lxml")
        links = []

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            
            # Convert relative URLs to absolute
            full_url = urljoin(current_url, href)
            
            # Only keep http/https URLs
            if not full_url.startswith(('http://', 'https://')):
                continue

            # Filter by domain if needed
            if same_domain_only and not self._is_same_domain(full_url):
                continue

            # Exclude Drive links at this stage
            if self._is_drive_link(full_url):
                continue

            # Remove fragments
            full_url = full_url.split('#')[0]
            
            links.append(full_url)

        return list(set(links))  # Remove duplicates

    def _find_drive_link_in_html(self, html: str, current_url: str) -> Optional[str]:
        """Find the first Google Drive link in HTML content."""
        soup = BeautifulSoup(html, "lxml")

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            full_url = urljoin(current_url, href)
            
            if self._is_drive_link(full_url):
                return full_url

        return None

    async def _download_from_drive(
        self, 
        session: aiohttp.ClientSession, 
        drive_url: str,
        filename: Optional[str] = None
    ) -> bool:
        """Download a file from Google Drive."""
        try:
            # Extract file ID
            file_id = self._extract_drive_file_id(drive_url)
            if not file_id:
                self.logger.warning(f"Could not extract file ID from: {drive_url}")
                self.failed_downloads.append({
                    "url": drive_url,
                    "error": "Could not extract file ID"
                })
                return False

            # Get direct download URL
            download_url = self._get_drive_download_url(file_id)
            
            self.logger.info(f"Downloading: {file_id}")

            # Download file
            async with session.get(
                download_url,
                timeout=aiohttp.ClientTimeout(total=300),  # 5 minutes
                allow_redirects=True
            ) as response:
                
                if response.status != 200:
                    # Handle large file warning
                    html = await response.text()
                    if 'download_warning' in html or 'virus scan warning' in html.lower():
                        soup = BeautifulSoup(html, 'lxml')
                        confirm_link = soup.find('a', {'id': 'uc-download-link'})
                        if confirm_link and confirm_link.get('href'):
                            confirm_url = 'https://drive.google.com' + confirm_link['href']
                            async with session.get(
                                confirm_url,
                                timeout=aiohttp.ClientTimeout(total=300),
                                allow_redirects=True
                            ) as confirm_response:
                                if confirm_response.status != 200:
                                    raise Exception(f"Failed: HTTP {confirm_response.status}")
                                file_content = await confirm_response.read()
                        else:
                            raise Exception(f"HTTP {response.status}")
                    else:
                        raise Exception(f"HTTP {response.status}")
                else:
                    file_content = await response.read()

                # Determine filename
                if not filename:
                    content_disposition = response.headers.get('Content-Disposition', '')
                    if 'filename=' in content_disposition:
                        filename = content_disposition.split('filename=')[1].strip('"\'')
                    else:
                        filename = f"{file_id}.pdf"

                # Ensure .pdf extension
                if not filename.lower().endswith('.pdf'):
                    filename += '.pdf'

                # Save file (avoid overwriting)
                filepath = self.output_dir / filename
                counter = 1
                original_filepath = filepath
                while filepath.exists():
                    filepath = original_filepath.with_stem(f"{original_filepath.stem}_{counter}")
                    counter += 1

                with open(filepath, 'wb') as f:
                    f.write(file_content)

                file_size_mb = len(file_content) / (1024 * 1024)
                self.logger.info(f"✓ Downloaded: {filename} ({file_size_mb:.2f} MB)")
                self.downloaded_files.append(str(filepath))
                return True

        except Exception as e:
            self.logger.error(f"✗ Failed to download {drive_url}: {str(e)}")
            self.failed_downloads.append({
                "url": drive_url,
                "error": str(e)
            })
            return False

    async def crawl_and_download(self, same_domain_only: bool = True) -> dict:
        """
        Main crawl method: Navigate 3 levels and download PDFs.
        
        Args:
            same_domain_only: Only follow links from the same domain
            
        Returns:
            Dictionary with crawl statistics
        """
        print(f"\n{'='*70}")
        print(f"Starting 3-Level Nested Drive Crawler")
        print(f"{'='*70}")
        if self.url_file:
            print(f"Reading from file: {self.url_file}")
            print(f"URLs to process: {len(self.main_urls)}")
            for i, url in enumerate(self.main_urls, 1):
                print(f"  {i}. {url}")
        else:
            print(f"Main URL: {self.main_url}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Same domain only: {same_domain_only}")
        print(f"{'='*70}\n")

        async with aiohttp.ClientSession(headers=config.DEFAULT_HEADERS) as session:
            
            # ===== LEVEL 1: Get links from main page(s) =====
            self.logger.info(f"🔍 LEVEL 1: Fetching {len(self.main_urls)} main page(s)...")
            
            for idx, main_url in enumerate(self.main_urls, 1):
                self.logger.info(f"Processing URL {idx}/{len(self.main_urls)}: {main_url}")
                
                main_html = await self._fetch_page(session, main_url)
                
                if not main_html:
                    self.logger.warning(f"❌ Failed to fetch: {main_url}")
                    continue

                links = self._extract_links(main_html, main_url, same_domain_only)
                self.level1_links.extend(links)
                
                self.logger.info(f"Found {len(links)} links on this page")
                await asyncio.sleep(config.REQUEST_DELAY)
            
            # Remove duplicates
            self.level1_links = list(set(self.level1_links))
            
            print(f"\n✓ Found {len(self.level1_links)} total unique links across all main pages")
            for i, link in enumerate(self.level1_links[:10], 1):
                print(f"  {i}. {link}")
            if len(self.level1_links) > 10:
                print(f"  ... and {len(self.level1_links) - 10} more")

            if not self.level1_links:
                print("\n❌ No links found on main pages!")
                return self._build_stats()

            # ===== LEVEL 2: Get sub-links from each Level 1 page =====
            print(f"\n🔍 LEVEL 2: Fetching sub-links from each page...")
            
            for level1_url in tqdm(self.level1_links, desc="Scanning Level 1 pages"):
                level1_html = await self._fetch_page(session, level1_url)
                
                if not level1_html:
                    self.logger.warning(f"Failed to fetch: {level1_url}")
                    continue

                sub_links = self._extract_links(level1_html, level1_url, same_domain_only)
                self.level2_links[level1_url] = sub_links
                
                if sub_links:
                    self.logger.info(f"Found {len(sub_links)} sub-links on {level1_url}")
                
                await asyncio.sleep(config.REQUEST_DELAY)

            total_level2_links = sum(len(links) for links in self.level2_links.values())
            print(f"\n✓ Found {total_level2_links} total sub-links across all pages")

            if total_level2_links == 0:
                print("\n❌ No sub-links found!")
                return self._build_stats()

            # ===== LEVEL 3: Find Drive links on each Level 2 page =====
            print(f"\n🔍 LEVEL 3: Finding Drive links on final pages...")
            
            all_level2_links = []
            for parent_url, sub_links in self.level2_links.items():
                all_level2_links.extend(sub_links)

            for level2_url in tqdm(all_level2_links, desc="Scanning Level 2 pages"):
                level2_html = await self._fetch_page(session, level2_url)
                
                if not level2_html:
                    continue

                drive_link = self._find_drive_link_in_html(level2_html, level2_url)
                
                if drive_link:
                    self.level3_drive_links[level2_url] = drive_link
                    self.logger.info(f"Found Drive link: {drive_link[:80]}...")
                
                await asyncio.sleep(config.REQUEST_DELAY)

            print(f"\n✓ Found {len(self.level3_drive_links)} Drive links")

            if not self.level3_drive_links:
                print("\n❌ No Drive links found!")
                return self._build_stats()

            # ===== DOWNLOAD: Download all PDFs from Drive links =====
            print(f"\n📥 DOWNLOADING: Downloading PDFs from Drive...")
            
            for source_url, drive_link in tqdm(
                self.level3_drive_links.items(), 
                desc="Downloading PDFs"
            ):
                await self._download_from_drive(session, drive_link)
                await asyncio.sleep(1)  # Be polite

        # Print summary
        return self._print_summary()

    def _build_stats(self) -> dict:
        """Build statistics dictionary."""
        return {
            "level1_links": len(self.level1_links),
            "level2_links": sum(len(links) for links in self.level2_links.values()),
            "drive_links_found": len(self.level3_drive_links),
            "files_downloaded": len(self.downloaded_files),
            "failed_downloads": len(self.failed_downloads),
            "downloaded_files": self.downloaded_files,
            "failed_downloads_list": self.failed_downloads
        }

    def _print_summary(self) -> dict:
        """Print crawl summary and return stats."""
        stats = self._build_stats()

        print(f"\n{'='*70}")
        print("✨ Crawl Complete!")
        print(f"{'='*70}")
        print(f"Level 1 links found: {stats['level1_links']}")
        print(f"Level 2 links found: {stats['level2_links']}")
        print(f"Drive links found: {stats['drive_links_found']}")
        print(f"Files downloaded: {stats['files_downloaded']}")
        print(f"Failed downloads: {stats['failed_downloads']}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"{'='*70}\n")

        if stats['failed_downloads'] > 0:
            print("❌ Failed downloads:")
            for failed in self.failed_downloads:
                print(f"  - {failed['url']}: {failed['error']}")

        return stats

    def run(self, same_domain_only: bool = True) -> dict:
        """Synchronous wrapper."""
        return asyncio.run(self.crawl_and_download(same_domain_only))


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="3-Level nested crawler: Main page -> Sub-pages -> Drive PDFs"
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="The main URL containing the initial list of links (or use -f for file)"
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to text file containing URLs (one per line)"
    )
    parser.add_argument(
        "-o", "--output",
        default="downloaded_pdfs",
        help="Output directory for downloaded PDFs (default: downloaded_pdfs)"
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent downloads (default: 5)"
    )
    parser.add_argument(
        "--all-domains",
        action="store_true",
        help="Follow links to other domains (default: same domain only)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.url and not args.file:
        parser.error("Either provide a URL or use -f/--file to specify a file containing URLs")

    crawler = NestedDriveCrawler(
        main_url=args.url,
        url_file=args.file,
        output_dir=args.output,
        concurrency=args.concurrency
    )

    stats = crawler.run(same_domain_only=not args.all_domains)

    # Exit with error if no files downloaded
    if stats['drive_links_found'] > 0 and stats['files_downloaded'] == 0:
        exit(1)


if __name__ == "__main__":
    main()

