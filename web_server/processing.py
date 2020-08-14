import cv2
import numpy as np
from werkzeug.datastructures import FileStorage
from tempfile import NamedTemporaryFile


def img2opencv(stream: FileStorage):
    # Based in https://stackoverflow.com/questions/54160208/how-to-use-opencv-in-python3-to-read-file-from-file-buffer/54162776#54162776
    return cv2.imdecode(np.fromstring(stream.read(), np.uint8), 1)

def videoCapture2array(video: cv2.VideoCapture):
    while(True): 
        grabbed, frame = video.read() 
        if not grabbed: break
        yield frame

def vid2opencv(stream: FileStorage):
    tempfile = NamedTemporaryFile(suffix='.mp4')
    stream.save(tempfile.name)
    video = cv2.VideoCapture(tempfile.name)
    tempfile.close()
    return videoCapture2array(video)
    

def crop(img):
    h, w = img.shape[:2]
    cy, cx = h//2-2, w//2-2
    m = min(h, w)//2
    return img[cy-m:cy+m, cx-m:cy+m]


def resize(img):
    dim = (254, 254)
    return cv2.resize(img, dim)


def crop_resize(img):
    cropped = crop(img)
    return resize(img)

def v_crop_resize(video):
    for frame in video:
        yield crop_resize(frame)