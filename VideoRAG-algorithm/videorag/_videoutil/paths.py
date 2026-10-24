import os


def video_name_from_path(video_path):
    return os.path.splitext(os.path.basename(video_path))[0]
