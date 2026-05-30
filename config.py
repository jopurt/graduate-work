# ===== Ollama API =====
OLLAMA_URL = "http://localhost:9117/api/generate"
MODEL_NAME = "mistral:latest"
# MODEL_NAME = "qwen3:4b"
# MODEL_NAME = "phi3:3.8b"

# ===== Video module parameters =====
MODEL_OPTIONS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 1024,
    "repeat_penalty": 1.2,
    "verbose": True,
    "seed": 42,
}

# Faster?
# MODEL_OPTIONS = {
#     "temperature": 0.3,
#     "top_p": 0.9,
#     "top_k": 20,
#     "num_predict": 512,
#     "repeat_penalty": 1.1,
#     "seed": 42,
#     "num_ctx": 4096,
# }

BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3

# JSONL_PATH = "result_objects/gatsby_mistral_2026-05-23_13-10.jsonl"
FRAMES_DIR = "frames/gatsby_test"
OUTPUT_DIR = "result_detections"

CONFIG_PATH = "GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "groundingdino_swint_ogc.pth"

# ===== Text segmentation =====
MAX_CHARS_IN_CHUNK = 25000
OVERLAP_CHARS_IN_TEXT = 200

BOOK_FILE_PATH = "gatsby.txt"
OUTPUT_FOLDER_PATH = "result_objects"

# ===== Frames parameters =====
VIDEO_FILE_PATH = "Velikiy.Getsby.2013_HDRip_dub_r5__[scarabey.org].avi"
FRAME_INTERVAL_SEC = 10.0

# ===== Instruction/rules =====
# INSTRUCTION = """
# You are a strict JSON extractor for **physical, tangible, man-made or natural objects** - such as clothing, furniture, vehicles, jewelry, buildings, tools, books, lamps, etc.
#
# Do NOT extract:
# - Body parts (face, hand, heart, eyes, hair, etc.)
# - Do NOT include character names in the "item" field
# - Abstract concepts (dream, hope, love, memory, etc.)
# - Weather or natural phenomena (wind, rain, moon, sky - unless described as a visual object)
# - Emotions or thoughts
# - Quoted speech or dialogue (e.g., "the longest day", "old sport", "I’ve never seen such beautiful shirts")
# - Titles, time references, or poetic phrases (e.g., "the longest day in the year", "that summer")
#
# ONLY extract objects that:
# - The object is physical, tangible, and distinct,
# - It is described with visual attributes (color, material, shape, etc.).
#
# Do NOT explain. Do NOT add commentary. Do NOT write prose.
# Only output valid JSON. If no valid object is described, return {{"item": null}}.
#
# Example input:
# "He took out a pile of shirts... checks in salmon and orange... sheer linen and thick silk..."
# Example output:
# {{
#   "item": "shirts",
#   "color": "salmon, orange, lavender, vivid green, red-and-navy-blue, purple-and-black, purple and gold",
#   "shape": "disordered, losing folds",
#   "texture": "sheer, thick, fine",
#   "material": "linen, silk, flannel",
#   "light": null,
#   "emotional_tone": "awe, sadness"
# }}
# """
#
# RULES = """
# Rules:
# 1) Output ONLY ONE object - the most vividly described.
# 2) NEVER invent details. If not in text - null.
# 3) Use ONLY these fields: item, color, shape, texture, material, light, emotional_tone.
# 4) The object must be a tangible, physical item - NOT a body part, person, or abstract idea.
# 5) If more than half of the fields would be null - return {{"item": null}}.
# 6) The output must be exclusively in json format.
# 7) The object is physical, tangible, distinct and inanimate
# 8) Do NOT use the names of the characters or name of book for objects
# 9) The name of the object should not be long
# """

INSTRUCTION = """
You are a strict JSON extractor for **physical, tangible, man-made or natural objects** - such as clothing, furniture, vehicles, jewelry, buildings, tools, books, lamps, etc.

CRITICAL: The "item" field must contain ONLY the name of the object itself, never a character's name or possessive form.
For example, from "Gatsby's mansion" extract "mansion", from "Daisy's dress" extract "dress".
Bad: {{"item": "Gatsby's mansion"}}
Good: {{"item": "mansion"}}

Do NOT extract:
- Body parts (face, hand, heart, eyes, hair, etc.)
- Abstract concepts (dream, hope, love, memory, etc.)
- Weather or natural phenomena (wind, rain, moon, sky - unless described as a visual object)
- Emotions or thoughts
- Quoted speech or dialogue (e.g., "the longest day", "old sport", "I’ve never seen such beautiful shirts")
- Titles, time references, or poetic phrases (e.g., "the longest day in the year", "that summer")

ONLY extract objects that:
- The object is physical, tangible, and distinct,
- It is described with visual attributes (color, material, shape, etc.).

Do NOT explain. Do NOT add commentary. Do NOT write prose.
Only output valid JSON. If no valid object is described, return {{"item": null}}.

Example input:
"He took out a pile of shirts... checks in salmon and orange... sheer linen and thick silk..."
Example output:
{{
  "item": "shirts",
  "color": "salmon, orange, lavender, vivid green, red-and-navy-blue, purple-and-black, purple and gold",
  "shape": "disordered, losing folds",
  "texture": "sheer, thick, fine",
  "material": "linen, silk, flannel",
  "light": null,
  "emotional_tone": "awe, sadness"
}}
"""

RULES = """
Rules:
1) Output ONLY ONE object - the most vividly described.
2) NEVER invent details. If not in text - null.
3) Use ONLY these fields: item, color, shape, texture, material, light, emotional_tone.
4) The object must be a tangible, physical item - NOT a body part, person, or abstract idea.
5) If more than half of the fields would be null - return {{"item": null}}.
6) The output must be exclusively in json format.
7) The object is physical, tangible, distinct and inanimate.
8) Remove any character name or possessive pronoun from the object name. 
   Extract only the common noun (e.g., "mansion" not "Gatsby's mansion", "car" not "his car").
9) The name of the object should be short (1-3 words).
"""
