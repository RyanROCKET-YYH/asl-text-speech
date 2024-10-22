import torch
import numpy as np
import pandas as pd
import cv2
from pytorch_i3d import InceptionI3d
import torch.nn as nn
import torch.nn.functional as nnf
import sys
import os
import django

sys.path.append('/afs/ece.cmu.edu/usr/hanqid/Public/asl-text-speech/app')     # change the path to your project path
os.environ['DJANGO_SETTINGS_MODULE'] = 'asl_text_speech.settings'  
django.setup()

from video.models import Video 

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

def process_sequence(frames):
    np_frames = np.asarray(frames, dtype=np.float32)
    r = np_frames[:,:,:,0]
    g = np_frames[:,:,:,1]
    b = np_frames[:,:,:,2]
    frames_reshaped = []
    frames_reshaped.append(r)
    frames_reshaped.append(g)
    frames_reshaped.append(b)
    np_frames_reshaped = np.asarray(frames_reshaped, dtype=np.float32)
    
    return torch.Tensor(np_frames_reshaped.reshape(1,3,np_frames_reshaped.shape[1],224,224))

def load_model(weights, num_classes):
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    return i3d

if __name__ == '__main__':

    num_classes = 100

    file = pd.read_csv("dataset/wlasl_class_list.txt", sep='\t', header=None)
    # file = pd.read_csv("D:\\asl-text-speech\\app\\video\WLASL_Inference\dataset\wlasl_class_list.txt", sep='\t', header=None)
    all_words = file[1].tolist()
    glosses = all_words[:num_classes]

    weights = 'weights/nslt_100.pt'
    # weights = 'D:\\asl-text-speech\\app\\video\WLASL_Inference\weights\\nslt_100.pt'
    i3d = load_model(weights, num_classes)

    sequence = []
    sentence = []
    threshold = 0.90
    frame_count = 0
    word = ""

    video_file_path = sys.argv[1]  # Get the video file path from the command-line arguments
    video_id = int(sys.argv[2])
    cap = cv2.VideoCapture(video_file_path)
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        if ret == 0:
            break
        output_frame = frame.copy()
        
        frame = pad(frame)
        frame = cv2.resize(frame, dsize=(256, 256))
        frame = center_crop(frame)
        
        frame = (frame / 255.) * 2 - 1

        sequence.append(frame)
        sequence = sequence[-40:]
        
        if len(sequence) == 40:
            per_frame_logits = i3d(process_sequence(sequence))
            predictions = torch.max(per_frame_logits, dim=2)[0]
            word = glosses[torch.argmax(predictions[0]).item()]

            if torch.max(nnf.softmax(predictions, dim=1)).item() > threshold: 
                if len(sentence) > 0: 
                    if word != sentence[-1]:
                        sentence.append(word)

                else:
                    sentence.append(word)

            
            frame_count += 1
            
    # Get the video instance and update its transcript field
    print(' '.join(sentence))
