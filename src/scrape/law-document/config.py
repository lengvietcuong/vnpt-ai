"""Configuration settings for the document scraper."""

import os

# Target URLs to scrape
TARGET_URLS = [
    "https://vanban.chinhphu.vn/he-thong-van-ban?classid=1&mode=1&typegroupid=1",
    "https://vanban.chinhphu.vn/he-thong-van-ban?classid=1&mode=1&typegroupid=2",
    "https://vanban.chinhphu.vn/he-thong-van-ban?classid=1&mode=1&typegroupid=3",
    "https://vanban.chinhphu.vn/he-thong-van-ban?classid=1&mode=1&typegroupid=4",
    "https://vanban.chinhphu.vn/he-thong-van-ban?classid=1&mode=1&typegroupid=5",
]

# Base directory (current script location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory paths
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Selenium settings
SELENIUM_HEADLESS = True  # Run browser in headless mode
SELENIUM_TIMEOUT = 30  # seconds
SELENIUM_IMPLICIT_WAIT = 10  # seconds
SELENIUM_PAGE_LOAD_TIMEOUT = 60  # seconds

# Wait times
TAB_CLICK_WAIT = 3  # seconds to wait after clicking a tab
PAGE_LOAD_WAIT = 2  # seconds to wait for page content to load
DOWNLOAD_DELAY = 1  # seconds between downloads

# Retry settings
MAX_DOWNLOAD_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

# Request settings
REQUEST_TIMEOUT = 60  # seconds for file downloads
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Supported file extensions
SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".doc",
    ".docx",
    ".rtf",
    ".odt",
]

# File naming
FILENAME_MAX_LENGTH = 100  # Maximum length for generated filenames

# Tab selectors (will be determined by inspecting the website)
# Common patterns for tabs in Vietnamese government sites
TAB_SELECTORS = [
    "ul.nav-tabs li a",
    "div.tabs a",
    "a[role='tab']",
    ".tab-list a",
    "ul.nav li a",
]

# Document link patterns (href attributes that likely contain documents)
DOCUMENT_LINK_PATTERNS = [
    "/download",
    "/portal/",
    ".pdf",
    ".doc",
    ".docx",
    ".rtf",
    ".odt",
]

# Pagination selectors
PAGINATION_SELECTORS = [
    "a.next",
    "a[rel='next']",
    "li.next a",
    "a[title*='Trang sau']",
    "a[title*='Next']",
]

# Elements to wait for
CONTENT_WAIT_SELECTORS = [
    "table",
    ".content",
    ".list",
    "div[class*='result']",
]

# Conversion settings
MARKDOWN_ADD_METADATA = True
MARKDOWN_ADD_TOC = False  # Add table of contents for long documents
MARKDOWN_PRESERVE_FORMATTING = True

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(LOG_DIR, "scraper.log")
ERROR_LOG_FILE = os.path.join(LOG_DIR, "errors.log")

# Performance
MAX_CONCURRENT_CONVERSIONS = 4  # Number of parallel document conversions
CHUNK_SIZE = 8192  # Bytes for downloading files

# Development/Debug settings
DEBUG_MODE = False
SAVE_PAGE_SOURCE = False  # Save HTML source for debugging
MAX_PAGES_PER_TAB = 0  # 0 = unlimited, set a number to limit for testing
