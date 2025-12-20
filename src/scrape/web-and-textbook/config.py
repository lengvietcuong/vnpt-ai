"""Configuration settings for the web scraper."""

# Request settings
REQUEST_TIMEOUT = 30  # seconds
REQUEST_DELAY = 1  # seconds between requests (be polite to servers)
MAX_RETRIES = 3

# User agent to identify the scraper
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Default headers for requests
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# HTML to Markdown settings
MARKDOWN_SETTINGS = {
    "body_width": 0,  # No line wrapping
    "ignore_links": False,
    "ignore_images": False,
    "ignore_emphasis": False,
    "ignore_tables": False,
    "single_line_break": False,
    "bypass_tables": False,
    "default_image_alt": "",
}

# Content selectors - common article containers (in order of preference)
ARTICLE_SELECTORS = [
    "article",
    "main",
    "[role='main']",
    ".post-content",
    ".article-content",
    ".entry-content",
    ".content",
    "#content",
    ".post",
    ".article",
    ".blog-post",
    ".story-body",
]

# Elements to remove from articles
ELEMENTS_TO_REMOVE = [
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "iframe",
    "noscript",
    ".advertisement",
    ".ad",
    ".ads",
    ".sidebar",
    ".comments",
    ".comment",
    ".social-share",
    ".share-buttons",
    ".related-posts",
    ".newsletter",
    ".popup",
    ".modal",
]

# File extensions to skip when crawling
SKIP_EXTENSIONS = [
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".tar", ".gz",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv",
    ".css", ".js", ".json", ".xml",
]

# URL patterns to skip
SKIP_URL_PATTERNS = [
    "/tag/",
    "/category/",
    "/author/",
    "/page/",
    "/login",
    "/register",
    "/cart",
    "/checkout",
    "#",
]
