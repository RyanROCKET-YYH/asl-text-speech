import torch
import numpy as np
import pandas as pd
import cv2
from pytorch_i3d import InceptionI3d
import torch.nn as nn
import torch.nn.functional as nnf
from torchinfo import summary

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
    i3d.load_state_dict(torch.load(weights))
    summary(i3d)
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    return i3d

if __name__ == '__main__':

    num_classes = 100

    file = pd.read_csv("dataset/wlasl_class_list.txt", sep='\t', header=None)
    all_words = file[1].tolist()
    glosses = all_words[:num_classes]

    weights = 'weights/nslt_100_006900_0.815436.pt'
    i3d = load_model(weights, num_classes)

    sequence = []
    sentence = []
    threshold = 0.85
    word = ""

    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('video.mp4')

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

                if len(sentence) > 5: 
                    sentence = sentence[-5:]
                
        cv2.rectangle(output_frame, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(output_frame, ' '.join(sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', output_frame)
        print(sentence, end='\r')

        # Break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()