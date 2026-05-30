from text_module import run_text_pipeline
from video_module import run_video_pipeline
from utils import extract_frames
import config


def main():
    extract_frames(config.VIDEO_FILE_PATH, config.FRAMES_DIR, config.FRAME_INTERVAL_SEC)

    run_text_pipeline()
    run_video_pipeline()


if __name__ == "__main__":
    main()
