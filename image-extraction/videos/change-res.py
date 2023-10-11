import os
import cv2

input_folder = './normal_res'
output_folder = './modified_res'

# Make sure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all video files in the input folder
video_files = [f for f in os.listdir(input_folder) if f.endswith('.MP4')]

for video_file in video_files:
    input_path = os.path.join(input_folder, video_file)
    output_path = os.path.join(output_folder, video_file)

    # Open the video file for reading
    video_capture = cv2.VideoCapture(input_path)

    # Get video properties
    frame_width = 640
    frame_height = 640
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Define a VideoWriter to save the modified video
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize the frame to 640x640 using INTER_CUBIC interpolation
        frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)

        out.write(frame)

    # Release the video capture and writer objects
    video_capture.release()
    out.release()

print("Videos processed and saved in the 'modified_res' folder.")
