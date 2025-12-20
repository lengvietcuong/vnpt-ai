import asyncio
import csv
import logging
from pathlib import Path

import aiohttp

from src.config import (
    INPUT_FILE_PATH,
    OUTPUT_FILE_PATH,
    MAX_REQUESTS_PER_SECOND,
    api_credentials,
)
from src.inference import process_questions
from src.utils import load_questions


logger = logging.getLogger(__name__)


async def main():
    logger.info("Running inference...")

    questions = load_questions(Path(INPUT_FILE_PATH))
    output_path = Path(OUTPUT_FILE_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(MAX_REQUESTS_PER_SECOND)

    async with aiohttp.ClientSession() as http_session:
        with open(output_path, "a", encoding="utf-8", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=["qid", "answer"])
            csv_writer.writeheader()
            csv_file.flush()
            logger.info(f"Created output CSV file at {output_path}")

            await process_questions(
                questions=questions,
                csv_file=csv_file,
                csv_writer=csv_writer,
                http_session=http_session,
                semaphore=semaphore,
                api_credentials=api_credentials,
            )
            return

    logger.info(f"Inference completed! Output saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
