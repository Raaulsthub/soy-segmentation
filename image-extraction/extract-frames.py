import os
import cv2

def get_video_names(folder_path: str):
    files = os.listdir(folder_path)
    video_files = [file for file in files if file.endswith('.MP4')]
    return video_files

def extract_frames(video_file: str, folder_path: str, img_folder: str, step: int):
    video_path = os.path.join(folder_path, video_file)
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if count % step == 0 and success:
            image = cv2.resize(image, (600, 600), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"./{img_folder}/{video_file[:-4]}_frame{count}.jpg", image)
        count += 1

def main():
    folder_path = './videos'
    for video_file in get_video_names(folder_path):
        extract_frames(video_file, folder_path, 'images', 20)


def len_images():
    print(len(os.listdir('./images')))



if __name__ == '__main__':
    main()
    len_images()

    