from utils import chunk_text, extract_visual_features
import json
import os
from datetime import datetime
import config


def run_text_pipeline():
    os.makedirs(config.OUTPUT_FOLDER_PATH, exist_ok=True)

    book_name = os.path.splitext(os.path.basename(config.BOOK_FILE_PATH))[0]
    model_short = config.MODEL_NAME.split(':')[0]

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

    os.makedirs(config.OUTPUT_FOLDER_PATH, exist_ok=True)

    output_file = f"{config.OUTPUT_FOLDER_PATH}/{book_name}_{model_short}_{date_str}.jsonl"

    chunks = chunk_text(
        config.BOOK_FILE_PATH,
        config.MAX_CHARS_IN_CHUNK,
        config.OVERLAP_CHARS_IN_TEXT
    )

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1}/{len(chunks)} ---")

        prompt_raw = f"""
        Now extract from this text:
        "{chunk}"
        """

        prompt = config.INSTRUCTION + config.RULES + prompt_raw + config.RULES

        response = extract_visual_features(prompt)

        # Save in JSONL
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(response, ensure_ascii=False) + '\n')

        print("LLM answer:", response)


if __name__ == "__main__":
    run_text_pipeline()
