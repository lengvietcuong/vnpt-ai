import logging
import os

from dotenv import load_dotenv

from src.models import APICredentials, LLMSettings, QuestionType


# Environment variables
load_dotenv()

VNPT_AI_AUTHORIZATION_TOKEN = os.environ["VNPT_AI_AUTHORIZATION_TOKEN"]
LLM_LARGE_TOKEN_ID = os.environ["LLM_LARGE_TOKEN_ID"]
LLM_LARGE_TOKEN_KEY = os.environ["LLM_LARGE_TOKEN_KEY"]

LLM_SMALL_TOKEN_ID = os.environ["LLM_SMALL_TOKEN_ID"]
LLM_SMALL_TOKEN_KEY = os.environ["LLM_SMALL_TOKEN_KEY"]

EMBEDDING_TOKEN_ID = os.environ["EMBEDDING_TOKEN_ID"]
EMBEDDING_TOKEN_KEY = os.environ["EMBEDDING_TOKEN_KEY"]

DB_HOST = os.environ["DB_HOST"]
DB_PORT = int(os.environ["DB_PORT"])
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

# File paths
INPUT_FILE_PATH = "/code/private_test.json"
OUTPUT_FILE_PATH = "/code/submission.csv"

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# API URLs
VNPT_AI_BASE_URL = "https://api.idg.vnpt.vn"
LLM_SMALL_ENDPOINT = "/data-service/v1/chat/completions/vnptai-hackathon-small"
LLM_LARGE_ENDPOINT = "/data-service/v1/chat/completions/vnptai-hackathon-large"
EMBEDDING_ENDPOINT = "/data-service/vnptai-hackathon-embedding"

# API Credentials
api_credentials = APICredentials(
    vnpt_ai_authorization_token=VNPT_AI_AUTHORIZATION_TOKEN,
    llm_large_token_id=LLM_LARGE_TOKEN_ID,
    llm_large_token_key=LLM_LARGE_TOKEN_KEY,
    llm_small_token_id=LLM_SMALL_TOKEN_ID,
    llm_small_token_key=LLM_SMALL_TOKEN_KEY,
    embedding_token_id=EMBEDDING_TOKEN_ID,
    embedding_token_key=EMBEDDING_TOKEN_KEY,
)

# API configurations
MAX_REQUESTS_PER_SECOND = 8
REQUEST_TIMEOUT_SECONDS = 30
TRY_AGAIN_DELAY_SECONDS = 1
MAX_ITERATIONS = 10

# Models
LLM_SMALL_MODEL = "vnptai_hackathon_small"
LLM_LARGE_MODEL = "vnptai_hackathon_large"
EMBEDDING_MODEL = "vnptai_hackathon_embedding"

# Model settings
TEMPERATURE = 0
MAX_TOKENS = 8192

SYSTEM_PROMPTS = {
    QuestionType.MATH: """You are the greatest mathematician to have ever lived.

When presented with a question, immediately write "```python" and use Python to help you find the answer. Write your reasoning steps in detail as code comments, then perform all the necessary calculations. Note that all math libraries are available for import (including sympy) and the output (not the final answer letter) must be printed using print().

After the code executes and you see the output, respond with the final answer letter wrapped in double curly braces (e.g., {{A}}). There is always one correct answer.""",
    QuestionType.READING: """You are the greatest reader to have ever lived.

When presented with a passage and a question, read and analyze the passage very carefully and thoroughly, while thinking aloud throughout.

For multiple-choice questions, you must wrap the final answer letter in double curly braces (e.g., {{A}}). There is always one correct answer.""",
    QuestionType.KNOWLEDGE: f"""You are the greatest scholar to have ever lived.

When presented with a question, use tools to look for the necessary information. First, search by keywords. If that doesn't work, try listing documents from the relevant topic(s), then listing sections in the relevant document(s), and finally reading the relevant section(s). Here are the available topics:
- Chính trị
- Chủ nghĩa
- Địa lý
- Hồ Chí Minh
- Hoá học
- Kinh tế
- Lịch sử
- Môi trường
- Pháp luật
- Sinh học
- Tâm lý học
- Tin học
- Tôn giáo
- Triết học
- Văn hoá
- Văn học
- Vật lý
- Y tế
Note that when using tools, the inputs must be exact.

If you cannot find anything at all, use your knowledge to answer the question to the best of your ability.

For multiple-choice questions, you must wrap the final answer letter in double curly braces (e.g., {{{{A}}}}). There is always one correct answer.""",
}

# Tools
KEYWORD_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "keyword_search",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of keywords for full-text search. Keywords must be as short (no more than 4 words) and unique as possible.",
                },
                "target_documents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of source documents .",
                },
            },
            "required": ["keywords"],
        },
    },
}

LIST_DOCUMENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_documents",
        "parameters": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of topics to retrieve documents from.",
                }
            },
            "required": ["topics"],
        },
    },
}

LIST_SECTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_sections",
        "parameters": {
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of Markdown documents to extract sections from.",
                }
            },
            "required": ["documents"],
        },
    },
}
MAX_HEADING_DEPTH = 3

READ_SECTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "read_sections",
        "parameters": {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of section headings to retrieve content for. The content of a section includes all its subsections.",
                },
                "documents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of Markdown documents to retrieve sections from.",
                },
            },
            "required": ["sections", "documents"],
        },
    },
}

TOOLS = {
    QuestionType.MATH: [],
    QuestionType.READING: [],
    QuestionType.KNOWLEDGE: [
        LIST_DOCUMENTS_TOOL,
        LIST_SECTIONS_TOOL,
        READ_SECTIONS_TOOL,
        KEYWORD_SEARCH_TOOL,
    ],
}

PREFILLS = {
    QuestionType.MATH: """```python\n""",
    QuestionType.READING: None,
    QuestionType.KNOWLEDGE: None,
}

LLM_SIZES = {
    QuestionType.MATH: "small",
    QuestionType.READING: "large",
    QuestionType.KNOWLEDGE: "small",
}

LLM_SETTINGS = {
    question_type: LLMSettings(
        llm_size=LLM_SIZES[question_type],
        system_prompt=SYSTEM_PROMPTS[question_type],
        tools=TOOLS[question_type],
        prefill=PREFILLS[question_type],
    )
    for question_type in QuestionType
}

# Python executor settings
PYTHON_EXECUTOR_TIMEOUT_SECONDS = 5
PYTHON_PATTERN = r"```python(.*?)```"

# Additional Settings
FALLBACK_ANSWER = "A"
LOG_RAW_RESPONSES = True

# RAG Settings
EMBEDDING_DIMENSIONS = 1024
SEMANTIC_SEARCH_TOP_K = 3
SEMANTIC_SEARCH_SIMILARITY_THRESHOLD = 0.6
KEYWORD_SEARCH_TOP_K = 10

# Knowledge ingestion
KNOWLEDGE_DATA_DIRECTORY = "data"
LAW_CSV_PATH = "data/Pháp luật.csv"

# Chunking Settings
MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
    ("######", "h6"),
]
MARKDOWN_CHUNK_SIZE = 1024
MARKDOWN_CHUNK_OVERLAP = 100
MINIMUM_CHUNK_CHARACTERS = 100

# Legal Document Abbreviations
MINISTRY_ABBREVIATIONS = {
    "BTC": "Bộ Tài chính",
    "BXD": "Bộ Xây dựng",
    "BKHCN": "Bộ Khoa học và Công nghệ",
    "BNNPTNT": "Bộ Nông nghiệp và Phát triển Nông thôn",
    "BNNPTTN": "Bộ Nông nghiệp và Phát triển Nông thôn",
    "BLĐTBXH": "Bộ Lao động – Thương binh và Xã hội",
    "NHNN": "Ngân hàng Nhà nước",
    "BGTVT": "Bộ Giao thông Vận tải",
    "BYT": "Bộ Y tế",
    "BGDĐT": "Bộ Giáo dục và Đào tạo",
    "BCA": "Bộ Công an",
    "BQP": "Bộ Quốc phòng",
    "BNG": "Bộ Ngoại giao",
    "BTP": "Bộ Tư pháp",
    "BTNMT": "Bộ Tài nguyên và Môi trường",
    "BNV": "Bộ Nội vụ",
    "BTTTT": "Bộ Thông tin và Truyền thông",
    "BKHĐT": "Bộ Kế hoạch và Đầu tư",
    "BVHTTDL": "Bộ Văn hóa, Thể thao và Du lịch",
    "BCT": "Bộ Công Thương",
}

DOCUMENT_TYPE_ABBREVIATIONS = {
    "QH": "Quốc hội",
    "TTLT": "Thông tư liên tịch",
    "TT": "Thông tư",
    "NĐ": "Nghị định",
    "QĐ": "Quyết định",
    "NQLT": "Nghị quyết liên tịch",
    "HP": "Hiến pháp",
}

ORGANIZATION_ABBREVIATIONS = {
    "CP": "Chính phủ",
    "TTg": "Thủ tướng",
    "UBTƯMTTQVN": "Ủy ban Trung ương Mặt trận Tổ quốc Việt Nam",
    "BHXH": "Bảo hiểm xã hội",
}

# Question classification settings
READING_MIN_LENGTH = 1000
READING_KEYWORD = "đoạn thông tin"
QUESTION_INDICATOR = "Câu hỏi: "
READING_CONTEXT_PREFIX = "Đoạn thông tin:"

# Reading passage chunking settings
READING_PASSAGE_PATTERN = r"--\s*(?:Đoạn văn|Document)\s+\d+\s*--"
MIN_READING_CHUNK_LENGTH = 150
MAX_READING_CHUNK_LENGTH = 2000
READING_SIMILARITY_THRESHOLD = 0.6
READING_TOP_K = 1

# Question classification patterns
LATEX_PATTERN = r"\$|\\(?:frac|sqrt|mu|theta|sigma|pi|alpha|beta|gamma|delta|epsilon|lambda|omega|times|div|sum|int|partial|infty|leq|geq|neq|approx)|\^"
PHYSICS_UNITS_PATTERN = r"\d+[.,]?\d*\s*(?:kg|cm|mm|km|kV|mV|μF|nF|pF|mA|μA|vg/ph|m/s|km/h|Hz|Pa|°C|AU|mét|độ)\b|\d+[.,]?\d*\s+(?:N|g|m|V|A|W|J|K|Ω)(?:\s|[,.]|$)"
CURRENCY_PATTERN = (
    r"(?:\d+[.,]?\d*\s*(?:tỷ|nghìn|triệu)?\s*(?:đô la|USD|usd)|\$\s*\d+[.,]?\d*)"
)
PERCENTAGE_PATTERN = r"\d+[.,]?\d*\s*%|phần trăm"
VARIABLE_PATTERN = r"\b[A-Z]\d+\b"
CHEMISTRY_PH_PATTERN = r"pH\s*=\s*\d+"
CHEMISTRY_MOL_PATTERN = r"\d+[.,]?\d*\s*mol"
DIGIT_PATTERN = r"\d"

PHYSICS_TERMS_PATTERN = r"điện trở|tụ điện|hiệu điện thế|điện áp|điện trường|từ trường|vận tốc|gia tốc|tốc độ|động lượng|năng lượng|công suất|áp suất|nhiệt độ|tần số|con lắc|quang học|khối lượng|trọng lượng|động cơ|quỹ đạo"
CHEMISTRY_TERMS_PATTERN = (
    r"nồng độ mol|khối lượng mol|phản ứng hóa học|cân bằng hóa học|công thức hóa học"
)
ECONOMICS_TERMS_PATTERN = (
    r"lãi suất|độ co giãn|chi phí biên|tỷ số hiện hành|tỷ số giá-thu nhập|vốn lưu động"
)
MATH_TERMS_PATTERN = r"phương trình|hàm số|véc-tơ|ma trận|đạo hàm|tích phân|tỷ lệ|bán kính|đường kính|chu vi|diện tích|thể tích"
CALCULATION_PATTERN = r"hãy tìm|hãy tính|đoạn mã|bao nhiêu cách|tính toán"
KNOWLEDGE_TERMS_PATTERN = r"lịch sử|văn hóa|chính trị|pháp luật|triết học|tôn giáo|văn học|nghệ thuật|xã hội học|tâm lý học|giáo dục|hồ chí minh|chính phủ"
LAW_REFERENCE_PATTERN = (
    r"điều \d+|quyết định số|nghị định số|thông tư số|/QĐ-|/NĐ-|/TT-"
)

REFUSE_PATTERN = r"tôi không thể|không thể trả lời|không thể cung cấp|không thể chia sẻ|từ chối trả lời|từ chối cung cấp|nằm ngoài phạm vi|không thuộc phạm vi|tôi là mô hình ngôn ngữ|hành vi vi phạm|trái pháp luật|không hỗ trợ"

# Miscellaneous
MAX_LOG_MESSAGE_LENGTH = 250
CAPITAL_A_ASCII_CODE = 65  # ASCII code for 'A'
