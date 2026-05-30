import config
import os
import glob
from pathlib import Path
from groundingdino.util.inference import load_model

from utils import load_items_from_jsonl, process_frame


def run_video_pipeline():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    files = glob.glob(os.path.join(config.OUTPUT_FOLDER_PATH, "*.jsonl"))
    jsonl_path = max(files, key=os.path.getmtime)

    items = load_items_from_jsonl(jsonl_path)
    if not items:
        print("Empty")
        return

    items_list = sorted(list(items))
    text_prompt = ".".join(items_list)
    print(f"Prompt for Grounding DINO:\n   {text_prompt}\n")

    model = load_model(config.CONFIG_PATH, config.CHECKPOINT_PATH)

    frame_paths = sorted(
        glob.glob(os.path.join(config.FRAMES_DIR, "*.jpg")) +
        glob.glob(os.path.join(config.FRAMES_DIR, "*.png"))
    )

    if not frame_paths:
        print(f"No frames in {config.FRAMES_DIR}")
        return

    print(f"Found {len(frame_paths)} frames")

    for i, frame_path in enumerate(frame_paths, 1):
        print(f"\n[{i}/{len(frame_paths)}] Processing: {Path(frame_path).name}")

        found_count = process_frame(
            model=model,
            image_path=frame_path,
            text_prompt=text_prompt,
            box_threshold=config.BOX_THRESHOLD,
            text_threshold=config.TEXT_THRESHOLD,
            output_dir=config.OUTPUT_DIR
        )

        print(f"Found objects: {found_count}")

    print(f"\nResults saved in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    run_video_pipeline()
