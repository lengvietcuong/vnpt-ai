"""Batch convert PDF files to Markdown using marker-pdf.
Since textbooks has complex structure this code use DL model to convert"""

import os
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import logging

try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
except ImportError:
    print("Error: marker-pdf is not installed.")
    print("Please install it with: pip install marker-pdf")
    exit(1)


class PDFToMarkdownConverter:
    """Convert PDF files to Markdown using marker-pdf."""

    def __init__(self, input_dir: str, output_dir: str, recursive: bool = False):
        """
        Initialize the converter.

        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save markdown files
            recursive: Whether to search recursively for PDFs
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.recursive = recursive
        self.converted_files: List[str] = []
        self.failed_files: List[Dict[str, str]] = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Check GPU availability
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.logger.info(f"🎮 GPU detected: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
            self.logger.info("Will use GPU acceleration for faster processing")
        else:
            self.logger.info("⚠️  No GPU detected - using CPU (this will be slower)")
            self.logger.info("To enable GPU: See ENABLE_GPU.md")

        # Load marker models once
        self.logger.info("Loading marker-pdf models (this may take a moment)...")
        try:
            self.model_dict = create_model_dict()
            self.converter = PdfConverter(
                artifact_dict=self.model_dict,
            )
            self.logger.info("Models loaded successfully!")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise

    def find_pdfs(self) -> List[Path]:
        """Find all PDF files in the input directory."""
        if self.recursive:
            pdf_files = list(self.input_dir.rglob("*.pdf"))
        else:
            pdf_files = list(self.input_dir.glob("*.pdf"))

        self.logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files

    def convert_pdf(self, pdf_path: Path) -> bool:
        """
        Convert a single PDF to markdown.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            True if conversion succeeded, False otherwise
        """
        try:
            self.logger.info(f"Converting: {pdf_path.name} (using {self.device.upper()})")

            # Convert PDF to markdown using marker-pdf v1.x API
            try:
                rendered = self.converter(str(pdf_path))
            except RuntimeError as e:
                # Handle CUDA out of memory errors
                if "out of memory" in str(e).lower():
                    self.logger.warning(f"GPU out of memory for {pdf_path.name}, retrying with CPU...")
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Recreate converter with CPU
                    self.converter = PdfConverter(artifact_dict=self.model_dict)
                    rendered = self.converter(str(pdf_path))
                else:
                    raise
            
            # Extract text from rendered output
            full_text, _, images = text_from_rendered(rendered)

            if not full_text or len(full_text.strip()) < 50:
                self.logger.warning(f"No content extracted from {pdf_path.name}")
                return False

            # Generate output filename
            relative_path = pdf_path.relative_to(self.input_dir)
            output_path = self.output_dir / relative_path.with_suffix('.md')

            # Create subdirectories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Get page count from rendered metadata
            page_count = len(rendered.pages) if hasattr(rendered, 'pages') else 'Unknown'

            # Add metadata header
            header = f"# {pdf_path.stem}\n\n"
            header += f"> Source: {pdf_path.name}\n"
            header += f"> Pages: {page_count}\n\n"
            header += "---\n\n"

            # Save markdown
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(header + full_text)

            # Save images if any
            if images:
                images_dir = output_path.parent / f"{pdf_path.stem}_images"
                images_dir.mkdir(exist_ok=True)
                for img_name, img_data in images.items():
                    img_path = images_dir / img_name
                    with open(img_path, 'wb') as img_file:
                        img_file.write(img_data)

            self.converted_files.append(str(output_path))
            self.logger.info(f"✓ Saved: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"✗ Failed to convert {pdf_path.name}: {str(e)}")
            self.failed_files.append({
                "file": str(pdf_path),
                "error": str(e)
            })
            return False

    def convert_all(self) -> Dict:
        """
        Convert all PDF files in the input directory.

        Returns:
            Dictionary with conversion statistics
        """
        pdf_files = self.find_pdfs()

        if not pdf_files:
            self.logger.warning("No PDF files found!")
            return {
                "total": 0,
                "converted": 0,
                "failed": 0,
                "converted_files": [],
                "failed_files": []
            }

        print(f"\n{'='*60}")
        print(f"Starting PDF to Markdown conversion")
        print(f"{'='*60}")
        print(f"Input directory: {self.input_dir.absolute()}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Total PDFs: {len(pdf_files)}")
        print(f"{'='*60}\n")

        # Convert each PDF with progress bar
        for pdf_path in tqdm(pdf_files, desc="Converting PDFs"):
            self.convert_pdf(pdf_path)

        # Print summary
        stats = {
            "total": len(pdf_files),
            "converted": len(self.converted_files),
            "failed": len(self.failed_files),
            "converted_files": self.converted_files,
            "failed_files": self.failed_files
        }

        print(f"\n{'='*60}")
        print("Conversion Complete!")
        print(f"{'='*60}")
        print(f"Total PDFs: {stats['total']}")
        print(f"Successfully converted: {stats['converted']}")
        print(f"Failed: {stats['failed']}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"{'='*60}\n")

        if stats['failed'] > 0:
            print("Failed files:")
            for failed in self.failed_files:
                print(f"  - {failed['file']}: {failed['error']}")

        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch convert PDF files to Markdown using marker-pdf."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "-o", "--output",
        default="markdown_output",
        help="Output directory for markdown files (default: markdown_output)"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search for PDFs recursively in subdirectories"
    )

    args = parser.parse_args()

    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        exit(1)

    # Create converter and run
    converter = PDFToMarkdownConverter(
        input_dir=args.input_dir,
        output_dir=args.output,
        recursive=args.recursive
    )

    stats = converter.convert_all()

    # Exit with error code if all conversions failed
    if stats['total'] > 0 and stats['converted'] == 0:
        exit(1)


if __name__ == "__main__":
    main()

