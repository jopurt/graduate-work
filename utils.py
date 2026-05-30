import requests
import config
import cv2
import os
import re
import json
import glob
from pathlib import Path
from groundingdino.util.inference import load_image, predict, annotate
from groundingdino.util import box_ops


def extract_visual_features(prompt):
    model_options = config.MODEL_OPTIONS.copy()

    data = {
        "model": config.MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": model_options
    }
    try:
        response = requests.post(config.OLLAMA_URL, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        return f"Ошибка: {e}"


def chunk_text(book_file, max_chars, overlap):
    with open(book_file, 'r', encoding='utf-8') as file:
        text = file.read().replace('\n', ' ').replace('\r', ' ').replace('-', '')

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # Find the nearest end of paragraph
        if end < len(text):
            para_end = text.find("\n\n", end)
            sent_end = text.find(".", end)
            if para_end != -1 and para_end < end + 1000:
                end = para_end + 2
            elif sent_end != -1 and sent_end < end + 500:
                end = sent_end + 1
            else:
                end = min(end, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def extract_frames(video_path, output_dir, interval_sec):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    # Calculate step in frames
    frame_step = int(fps * interval_sec)
    if frame_step < 1:
        frame_step = 1

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_step == 0:
            timestamp = frame_count / fps if fps > 0 else saved_count * interval_sec
            frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}_{timestamp:06.2f}s.jpg")
            cv2.imwrite(frame_path, frame)

            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Извлечено {saved_count} кадров из {total_frames} (интервал: {interval_sec} сек, длительность: {duration_sec:.0f} сек)")


def get_latest_jsonl(output_dir: str):
    files = glob.glob(os.path.join(output_dir, "*.jsonl"))

    if not files:
        return None

    return max(files, key=os.path.getmtime)


def clean_item_name(item: str) -> str:
    if not item or item.lower() in ["null", "none", ""]:
        return ""

    cleaned = item.replace("'", "").replace("`", "").replace('"', '')

    cleaned = re.sub(r"[^a-zA-Z0-9\s\-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()

    return cleaned if len(cleaned) >= 2 else ""


def load_items_from_jsonl(jsonl_path: str) -> list[str]:
    items = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)

                if isinstance(obj, str):
                    obj = json.loads(obj)

                item = obj.get("item")
                cleaned = clean_item_name(item)
                if cleaned and cleaned not in items:
                    items.append(cleaned)
            except json.JSONDecodeError as e:
                print(e)
                continue

    print(f"Items:{items}")
    return items


def process_frame(model, image_path, text_prompt, box_threshold, text_threshold, output_dir):
    try:
        image_source, image = load_image(image_path)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        count = len(boxes)
        if count > 0:
            for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                xyxy = box_ops.box_cxcywh_to_xyxy(box).tolist()
                print(f" {phrase} (confidence: {logit:.3f}, bbox: {[round(v, 3) for v in xyxy]})")

            annotated_frame = annotate(
                image_source=image_source,
                boxes=boxes,
                logits=logits,
                phrases=phrases
            )

            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, Path(image_path).name)
            cv2.imwrite(out_path, annotated_frame)
            print(f"Сохранено: {out_path}")
        else:
            print("Ничего не найдено (пропуск)")

        return count

    except Exception as e:
        print(e)
        return 0
