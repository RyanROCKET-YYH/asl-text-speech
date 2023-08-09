import sys
from socket import socket, AF_INET, SOCK_DGRAM, SOCK_STREAM
import cv2
import time
import pickle
import numpy as np

def pad(img):
    max_size = max(img.shape)
    top = int((max_size-img.shape[0])/2)
    bot = int((max_size-img.shape[0])/2)
    left = int((max_size-img.shape[1])/2)
    right = int((max_size-img.shape[1])/2)
    return cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT, None, value = 0)

def center_crop(img):
    center = img.shape
    w = 224
    h = 224
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2

    return img[int(y):int(y+h), int(x):int(x+w)]


def send_frame(data, prev, flag):
    host = 'localhost'
    server = (host, 4000)
    sock = socket(AF_INET, SOCK_STREAM)
    sock.connect(server)

    if (flag == 0):
        data_string = bytes(str(0), 'utf-8')
    else:
        data = cv2.imencode('.jpg', data)[1]
        data_string = np.array(data).tobytes()

    flag = bytes(str(flag), 'utf-8')
    try:
        sock.sendall(flag)
        data = sock.recv(1024)
        if data != ' ' and data != prev:
            print(sys.stderr, data)

        # print(sys.stderr, 'sending data')
        sock.sendall(data_string)
    finally:
        # print(sys.stderr, 'closing socket')
        sock.close()

    return data


start = time.time()

# cap = cv2.VideoCapture(0)
prev_word = " "
cap = cv2.VideoCapture("yep.mp4")

while cap.isOpened():

    # Read feed
    ret, frame = cap.read()

    if ret == 0:
        prev_word = send_frame(frame, prev_word, 0)
        break

    output_frame = frame.copy()

    frame = pad(frame)
    frame = cv2.resize(frame, dsize=(256, 256))
    frame = center_crop(frame)

    prev_word = send_frame(frame, prev_word, 1)

    cv2.imshow('frame', output_frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

end = time.time()
print('Time spent: ', end - start)