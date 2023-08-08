import asyncio
import websockets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from pytorch_i3d import InceptionI3d
import cv2
import pandas as pd
import base64

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
    r = np_frames[:, :, :, 0]
    g = np_frames[:, :, :, 1]
    b = np_frames[:, :, :, 2]
    frames_reshaped = []
    frames_reshaped.append(r)
    frames_reshaped.append(g)
    frames_reshaped.append(b)
    np_frames_reshaped = np.asarray(frames_reshaped, dtype=np.float32)
    return torch.Tensor(np_frames_reshaped.reshape(1, 3, np_frames_reshaped.shape[1], 224, 224))

def load_model(weights, num_classes):
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()
    return i3d

async def handle_client(websocket, path):
    print(f"Client connected from {websocket.remote_address}")

    # Load the model and other setup tasks
    num_classes = 100
    weights = '/afs/ece.cmu.edu/usr/hanqid/Public/asl-text-speech/WLASL_Inference/weights/nslt_100.pt'
    i3d = load_model(weights, num_classes)
    file = pd.read_csv("/afs/ece.cmu.edu/usr/hanqid/Public/asl-text-speech/WLASL_Inference/dataset/wlasl_class_list.txt", sep='\t', header=None)
    all_words = file[1].tolist()
    glosses = all_words[:num_classes]

    full_data = bytearray()
    frame_count = 0
    sequence = []
    threshold = 0.90
    transcript = []
    try: 
        while True:
            if len(transcript)>0:
                message = transcript[-1]
            else:
                message = ' '
            if len(message) > 2:
                await websocket.send(message)

            data = await websocket.recv()
            if not data:
                break

            full_data += data.encode()  # Convert string data to bytes and append to the buffer
            print('Receiving frame')

            # Try to decode the received data as an image
            try:
                # Decode Base64 data to raw bytes
                if full_data.startswith(b'data:image/jpeg;base64,'):
                    base64_data = full_data[len(b'data:image/jpeg;base64,'):]
                    decoded_data = base64.b64decode(base64_data)
                    np_data = np.frombuffer(decoded_data, dtype='uint8')
                else:
                    # If the data is not in the expected format, skip this frame
                    continue
                frame = ((cv2.imdecode(np_data, cv2.IMREAD_COLOR)) / 225) * 2 - 1

                if frame is not None:
                    # Process the frame
                    frame = pad(frame)
                    frame = cv2.resize(frame, dsize=(256, 256))
                    frame = center_crop(frame)

                    sequence.append(frame)

                    # Make predictions
                    if len(sequence) == 40 and frame_count%4 == 0:
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

                        # Send the predicted word back to the client
                        #await websocket.send(word)
                    frame_count+=1
                    full_data = bytearray()  # Clear the buffer for the next frame
            except cv2.error as e:
                print(f"Error processing frame: {e}")
            except Exception as e:
                print(f"Error occurred: {e}")
    except websockets.ConnectionClosedError:
        print(f"Connection closed by {websocket.remote_address}")
    except Exception as e:
        print(f"Error occurred: {e}")

async def main():
    # Start the WebSocket server
    async with websockets.serve(handle_client, '172.19.138.16', 4000):
        print('WebSocket server started.')
        await asyncio.Future()  # Run forever

# Run the main event loop
if __name__ == '__main__':
    asyncio.run(main())
