from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM, SOCK_STREAM
import sys
import time
import pickle
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from pytorch_i3d import InceptionI3d
import torch.nn as nn
import json
import torch.nn.functional as nnf
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    return i3d

if __name__ == '__main__':
    num_classes = 100

    file = pd.read_csv("/afs/ece.cmu.edu/usr/hanqid/Public/asl-text-speech/WLASL_Inference/dataset/wlasl_class_list.txt", sep='\t', header=None)
    # file = pd.read_csv("D:\\asl-text-speech\\app\\video\WLASL_Inference\dataset\wlasl_class_list.txt", sep='\t', header=None)
    all_words = file[1].tolist()
    glosses = all_words[:num_classes]

    weights = '/afs/ece.cmu.edu/usr/hanqid/Public/asl-text-speech/WLASL_Inference/weights/nslt_100.pt'
    # weights = 'D:\\asl-text-speech\\app\\video\WLASL_Inference\weights\\nslt_100.pt'
    i3d = load_model(weights, num_classes)

    full_data = bytearray()
    frame_count = 0
    sequence = []
    transcript = []
    threshold = 0.90
    word = ""

    host = '172.19.138.16' #Server ip
    port = 4000

    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind((host, port))

    # Listen for incoming connections
    sock.listen(1)

    while True:
        # Wait for a connection
        print(sys.stderr, 'waiting for a connection')
        connection, client_address = sock.accept()
        
        try:
            #print(sys.stderr, 'connection from', client_address)
            stream_flag = connection.recv(16)
            stream_flag = stream_flag.decode()
            
            if stream_flag == '0':
                break
            
            if len(transcript)>0:
                message = transcript[-1]
            else:
                message = ' '
                
            connection.sendall(bytes(message, 'utf-8'))
            #Receive data
            while True:
                data = connection.recv(800000)
                
                if data:
                    full_data.extend(data)
                    print(sys.stderr, 'Receiving frame')
                else:
                    print(sys.stderr, 'Finish receiving frame', client_address)
                    if stream_flag != 0:
                        full_data = np.frombuffer(full_data, dtype='uint8')
                        sequence += [((cv2.imdecode(full_data, cv2.IMREAD_COLOR))/225)*2 - 1]
                
                    #Make predictions
                    if len(sequence) == 40:
                        per_frame_logits = i3d(process_sequence(sequence))
                        predictions = torch.max(per_frame_logits, dim=2)[0]
                        word = glosses[torch.argmax(predictions[0]).item()]
                        if torch.max(nnf.softmax(predictions, dim=1)).item() > threshold: 
                            if len(transcript) > 0: 
                                if word != transcript[-1]:
                                    transcript.append(word)
                            else:
                                transcript.append(word)
                        sequence = sequence[-39:]

                    frame_count += 1
                    full_data = bytearray()
                    break

        finally:
            #Close connections
            connection.close()
            if stream_flag == '0':
                break
                
    cv2.destroyAllWindows()