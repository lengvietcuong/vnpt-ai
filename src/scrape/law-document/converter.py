"""Document to Markdown converter supporting multiple formats."""

import os
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

import fitz  # PyMuPDF
from docx import Document
import pypandoc

import config


# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.ERROR_LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DocumentConverter:
    """Converts various document formats to markdown."""

    def __init__(self):
        """Initialize the converter."""
        self.supported_formats = {
            '.pdf': self._convert_pdf,
            '.docx': self._convert_docx,
            '.doc': self._convert_doc,
            '.rtf': self._convert_rtf,
            '.odt': self._convert_odt,
        }

    def convert_file(self, file_path: str, metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Convert a document file to markdown.

        Args:
            file_path: Path to the document file
            metadata: Optional metadata dict with keys: source_url, typegroupid, tab_name

        Returns:
            Markdown content as string, or None if conversion failed
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in self.supported_formats:
            logger.error(f"Unsupported file format: {file_ext}")
            return None

        try:
            logger.info(f"Converting {file_path} to markdown...")
            converter_func = self.supported_formats[file_ext]
            markdown_content = converter_func(file_path)
            
            if not markdown_content or len(markdown_content.strip()) < 10:
                logger.warning(f"Conversion resulted in empty or very short content: {file_path}")
                return None

            # Add metadata header if enabled
            if config.MARKDOWN_ADD_METADATA and metadata:
                markdown_content = self._add_metadata_header(markdown_content, file_path, metadata)

            return markdown_content

        except Exception as e:
            logger.error(f"Failed to convert {file_path}: {str(e)}", exc_info=True)
            return None

    def _convert_pdf(self, file_path: str) -> str:
        """Convert PDF to markdown using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            markdown_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text with formatting
                text = page.get_text("text")
                
                if text.strip():
                    # Add page separator for multi-page documents
                    if page_num > 0:
                        markdown_parts.append(f"\n---\n\n**Page {page_num + 1}**\n\n")
                    
                    # Clean up the text
                    text = self._clean_text(text)
                    markdown_parts.append(text)

            doc.close()
            
            return "\n\n".join(markdown_parts)

        except Exception as e:
            logger.error(f"PDF conversion error: {str(e)}")
            raise

    def _convert_docx(self, file_path: str) -> str:
        """Convert DOCX to markdown using python-docx."""
        try:
            doc = Document(file_path)
            markdown_parts = []

            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue

                # Check paragraph style to determine markdown formatting
                style = paragraph.style.name.lower()
                
                if 'heading 1' in style or style == 'title':
                    markdown_parts.append(f"# {text}")
                elif 'heading 2' in style:
                    markdown_parts.append(f"## {text}")
                elif 'heading 3' in style:
                    markdown_parts.append(f"### {text}")
                elif 'heading 4' in style:
                    markdown_parts.append(f"#### {text}")
                elif 'heading 5' in style:
                    markdown_parts.append(f"##### {text}")
                elif 'heading 6' in style:
                    markdown_parts.append(f"###### {text}")
                else:
                    # Regular paragraph
                    markdown_parts.append(text)

            # Process tables
            for table in doc.tables:
                markdown_parts.append(self._convert_table_to_markdown(table))

            return "\n\n".join(markdown_parts)

        except Exception as e:
            logger.error(f"DOCX conversion error: {str(e)}")
            raise

    def _convert_doc(self, file_path: str) -> str:
        """Convert DOC to markdown using pypandoc."""
        try:
            # pypandoc requires pandoc to be installed on the system
            markdown = pypandoc.convert_file(file_path, 'md', format='doc')
            return self._clean_text(markdown)
        except Exception as e:
            logger.error(f"DOC conversion error: {str(e)}")
            logger.warning("Make sure pandoc is installed on your system: https://pandoc.org/installing.html")
            raise

    def _convert_rtf(self, file_path: str) -> str:
        """Convert RTF to markdown using pypandoc."""
        try:
            markdown = pypandoc.convert_file(file_path, 'md', format='rtf')
            return self._clean_text(markdown)
        except Exception as e:
            logger.error(f"RTF conversion error: {str(e)}")
            logger.warning("Make sure pandoc is installed on your system: https://pandoc.org/installing.html")
            raise

    def _convert_odt(self, file_path: str) -> str:
        """Convert ODT to markdown using pypandoc."""
        try:
            markdown = pypandoc.convert_file(file_path, 'md', format='odt')
            return self._clean_text(markdown)
        except Exception as e:
            logger.error(f"ODT conversion error: {str(e)}")
            logger.warning("Make sure pandoc is installed on your system: https://pandoc.org/installing.html")
            raise

    def _convert_table_to_markdown(self, table) -> str:
        """Convert a docx table to markdown format."""
        markdown_lines = []
        
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip() for cell in row.cells]
            markdown_lines.append("| " + " | ".join(cells) + " |")
            
            # Add header separator after first row
            if i == 0:
                markdown_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
        
        return "\n".join(markdown_lines)

    def _clean_text(self, text: str) -> str:
        """Clean up text by removing excessive whitespace and normalizing."""
        # Remove excessive blank lines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        # Normalize Unicode characters
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        return text.strip()

    def _add_metadata_header(self, content: str, file_path: str, metadata: Dict) -> str:
        """Add metadata header to the markdown content."""
        filename = Path(file_path).name
        file_ext = Path(file_path).suffix
        
        header_lines = [
            "---",
            f"**Document:** {filename}",
            f"**Format:** {file_ext.upper().replace('.', '')}",
            f"**Converted:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        if metadata.get('source_url'):
            header_lines.append(f"**Source URL:** {metadata['source_url']}")
        
        if metadata.get('typegroupid'):
            header_lines.append(f"**Type Group ID:** {metadata['typegroupid']}")
        
        if metadata.get('page_num'):
            header_lines.append(f"**Page Number:** {metadata['page_num']}")
        
        if metadata.get('original_text'):
            header_lines.append(f"**Original Title:** {metadata['original_text'][:100]}")
        
        header_lines.append("---")
        header_lines.append("")  # Empty line before content
        
        return "\n".join(header_lines) + "\n\n" + content

    def save_markdown(self, markdown_content: str, output_path: str) -> bool:
        """
        Save markdown content to a file.

        Args:
            markdown_content: The markdown string to save
            output_path: Path where to save the file

        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Saved markdown: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save markdown to {output_path}: {str(e)}")
            return False

    def convert_and_save(self, input_file: str, output_file: str, metadata: Optional[Dict] = None) -> bool:
        """
        Convert a document and save it as markdown.

        Args:
            input_file: Path to input document
            output_file: Path where to save markdown
            metadata: Optional metadata dict

        Returns:
            True if successful, False otherwise
        """
        markdown_content = self.convert_file(input_file, metadata)
        
        if markdown_content:
            return self.save_markdown(markdown_content, output_file)
        
        return False


def main():
    """Test the converter with a sample file."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert documents to markdown")
    parser.add_argument("input_file", help="Path to input document")
    parser.add_argument("-o", "--output", help="Output markdown file path")
    parser.add_argument("--url", help="Source URL for metadata")
    
    args = parser.parse_args()

    converter = DocumentConverter()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input_file)
        output_path = input_path.with_suffix('.md')

    # Convert
    metadata = {}
    if args.url:
        metadata['source_url'] = args.url

    success = converter.convert_and_save(args.input_file, output_path, metadata)
    
    if success:
        print(f"✓ Successfully converted to: {output_path}")
    else:
        print(f"✗ Conversion failed")
        exit(1)


if __name__ == "__main__":
    main()
