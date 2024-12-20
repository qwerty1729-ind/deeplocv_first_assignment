import cv2
from yolov5 import detect

def detect_objects_on_video(video_path, output_path, weights_path='yolov5s.pt', conf_threshold=0.25):
    """
    Run YOLOv5 detection on a video and save the output.
    """
    # Run YOLO detection directly on the video file
    detect.run(
        weights=weights_path,
        source=video_path,  # Pass video file path directly
        conf_thres=conf_threshold,
        save_txt=False,
        save_crop=False,
        project=output_path,  # Save output in a folder
        exist_ok=True
    )
    print(f"Detection complete. Output saved at: {output_path}")

# Example usage
if __name__ == "__main__":
    input_video = "sample_video.mp4"  # Replace with the path to your sample video
    output_folder = "yolo_output"     # Folder to save results
    detect_objects_on_video(input_video, output_folder)
