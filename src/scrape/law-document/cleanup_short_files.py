"""Cleanup script to remove markdown files with too few lines."""

import os
import logging
from pathlib import Path

import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, 'cleanup.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def count_lines(file_path: str) -> int:
    """Count the number of lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return 0


def cleanup_short_files(min_lines: int = 20, dry_run: bool = False):
    """
    Remove markdown files that have fewer than min_lines lines.
    
    Args:
        min_lines: Minimum number of lines required (default: 20)
        dry_run: If True, only report what would be deleted without actually deleting
    """
    output_dir = Path(config.OUTPUT_DIR)
    
    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return
    
    logger.info(f"{'DRY RUN: ' if dry_run else ''}Scanning {output_dir} for files with less than {min_lines} lines")
    logger.info("=" * 60)
    
    deleted_count = 0
    kept_count = 0
    total_size_deleted = 0
    
    # Find all markdown files
    md_files = list(output_dir.glob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files")
    
    for md_file in md_files:
        line_count = count_lines(str(md_file))
        file_size = md_file.stat().st_size
        
        if line_count < min_lines:
            if dry_run:
                logger.info(f"Would delete: {md_file.name} ({line_count} lines, {file_size} bytes)")
            else:
                try:
                    md_file.unlink()
                    logger.info(f"Deleted: {md_file.name} ({line_count} lines, {file_size} bytes)")
                    total_size_deleted += file_size
                except Exception as e:
                    logger.error(f"Failed to delete {md_file.name}: {e}")
            deleted_count += 1
        else:
            logger.debug(f"Keeping: {md_file.name} ({line_count} lines)")
            kept_count += 1
    
    logger.info("=" * 60)
    logger.info(f"Summary:")
    logger.info(f"  Files deleted: {deleted_count}")
    logger.info(f"  Files kept: {kept_count}")
    if not dry_run:
        logger.info(f"  Total size deleted: {total_size_deleted / 1024:.2f} KB")
    logger.info(f"  {'DRY RUN - no files were actually deleted' if dry_run else 'Cleanup complete'}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Remove markdown files with too few lines"
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=20,
        help="Minimum number of lines required (default: 20)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    args = parser.parse_args()
    
    cleanup_short_files(min_lines=args.min_lines, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
