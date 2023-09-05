import os
import tempfile
import subprocess
from ultralytics import YOLO
import cv2
import shutil

# Video path and model
m3u8_url = 'https://api.forzify.com/eliteserien/playlist.m3u8/6526:1930000:1990000/Manifest.m3u8'
output_video_path = '//home/martin/Sep04/BoundingBox/YOLO-PredictBOX/Vbestm.avi'
model = YOLO('/home/martin/Sep04/BoundingBox/bestm.pt')
skip_n_frames = 1  # process every skip_n_frames frame
frame_count = 0

# Define class names (replace with actual names if available)
class_names = ['player', 'ball', 'logo']

# Download the m3u8 file using ffmpeg
temp_video_path = 'temp_video.mp4'
subprocess.call(['ffmpeg', '-i', m3u8_url, '-c', 'copy', temp_video_path])

# Open video file
video = cv2.VideoCapture(temp_video_path)

# Get video properties
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Temporary directory to save frames
temp_dir = 'tmp/'
os.makedirs(temp_dir, exist_ok=True)

try:
    while True:
        # Read the next frame from the video
        ret, frame = video.read()
        # Break the loop if we have reached the end of the video
        if not ret:
            break
        frame_count += 1
        # Skip frames
        if frame_count % skip_n_frames != 0:
            continue

        # Save the frame to a temporary file
        frame_path = os.path.join(temp_dir, 'temp_frame.png')
        cv2.imwrite(frame_path, frame)

        # Run the model on the frame
        results = model(frame_path)
        # Assuming results is a tensor of shape [number_of_boxes, 6]
        # where each row represents [x1, y1, x2, y2, confidence, class]
        # Check the type of results and obtain the bounding boxes
        results = model(frame_path)
        for result in results:
           boxes = result.boxes.data  # or however you access the boxes

        


        # Iterate through each box and draw it on the frame
        for box in boxes:
            
            x1, y1, x2, y2, conf, cls= box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Draw the rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Optionally, add a label with class and confidence
            if 0 <= int(cls) < len(class_names):
                label = f'{class_names[int(cls)]}: {conf:.2f}'
            else:
                print(f"Warning: Invalid class index {cls}")
                label = f'Unknown: {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame with bounding boxes to the output video
        out.write(frame)
finally:
    # Cleanup
    shutil.rmtree(temp_dir)
    os.remove(temp_video_path)  # remove the temporary video file

    # Release the video file pointers
    video.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()