import cv2
import numpy as np
import os
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(63, 1)))

model.add(Conv1D(64, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dense(38, activation='softmax'))

weight_file = 'alphabets1.keras'
model.load_weights(weight_file)


def mediapipe_detection(input_image, input_model):
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image.flags.writeable = False
    output_results = input_model.process(input_image)
    input_image.flags.writeable = True
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    return input_image, output_results


def draw_landmarks(input_image, input_results):
    mp_drawing.draw_landmarks(input_image, input_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 122), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(input_image, input_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 117, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 66, 122), thickness=2, circle_radius=2))


def extract_keypoints(input_result):
    x = np.array([[res.x] for res in
                  input_result.right_hand_landmarks.landmark]).flatten() if input_result.right_hand_landmarks else np.zeros(63)
    y = np.array([[res.y] for res in
                  input_result.right_hand_landmarks.landmark]).flatten() if input_result.right_hand_landmarks else np.zeros(63)
    z = np.array([[res.z] for res in
                  input_result.right_hand_landmarks.landmark]).flatten() if input_result.right_hand_landmarks else np.zeros(63)
    range_x = max(x) - min(x)
    range_y = max(y) - min(y)
    range_z = max(z) - min(z)
    rh = np.array([[(res.x - min(x)) / range_x, (res.y - min(y)) / range_y, (res.z - min(z)) / range_z] for res in
                   input_result.right_hand_landmarks.landmark]).flatten() if input_result.right_hand_landmarks else np.zeros(63)
    return rh


def extract_keypoints_left(results):
    x = np.array([[res.x] for res in
                  results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    y = np.array([[res.y] for res in
                  results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    z = np.array([[res.z] for res in
                  results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    range_x = max(x) - min(x)
    range_y = max(y) - min(y)
    range_z = max(z) - min(z)
    rh = np.array([[(res.x - min(x)) / range_x, (res.y - min(y)) / range_y, (res.z - min(z)) / range_z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    return rh


import collections


def distance(w1, w2):
    l1 = len(w1)
    l2 = len(w2)
    if l1 == 0:
        return l2
    if l2 == 0:
        return l1
    cost = [[0] * l2 for _ in range(l1)]
    res = [[0] * l2 for _ in range(l1)]
    for i in range(l1):
        for j in range(l2):
            if w1[i] != w2[j]:
                cost[i][j] = 1.01  # priority: add/delete > change   e.g. helo -> hello
                # cost[i][j] = 1  # priority: change > add/delete   e.g. helo -> help
    for i in range(l1):
        for j in range(l2):
            if i >= 1:
                a = res[i - 1][j] + 1
            else:
                a = 99
            if j >= 1:
                b = res[i][j - 1] + 1
            else:
                b = 99
            if (i >= 1) and (j >= 1):
                c = res[i - 1][j - 1] + cost[i][j]
            else:
                c = 99
            if (i == 0) and (j == 0):
                if w1[i] != w2[j]:
                    res[i][j] = 1
            else:
                res[i][j] = min(a, b, c)
    return res[l1 - 1][l2 - 1]


def correct(input, tol=2):
    input = input.lower()
    output = input
    if output.isnumeric() or len(output) == 1:
        return output.upper()
    for i in range(len(input)):
        if input[i] == '6':
            output = input[:i] + 'w' + input[i + 1:]
        if input[i] == '9':
            output = input[:i] + 'f' + input[i + 1:]
    if output not in word_num:
        for i in range(1, tol + 1):
            for word in word_num:
                if distance(word, input) <= i:
                    output = word
                    break
            if output != input:
                break
    return output.upper()


word_num = collections.defaultdict(int)
with open("lemma.num", "r") as file:
    for line in file:
        parts = line.split()
        word_num[parts[2].lower()] = parts[1]

wrong_word = "6orld"
print(correct(wrong_word))
wrong_word = "helo"
print(correct(wrong_word))

word_list = []
with open("lemma.num", "r") as file:
    for line in file:
        parts = line.split()
        word_list.append(parts[2])


def associat(input):
    input = input.lower()
    for word in word_list:
        if word[:len(input)] == input:
            return word.upper()
    return input.upper()


cap = cv2.VideoCapture(0)  # cap is the webcam
# access the mediapipe model as holistic
alphabets = np.array(
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
     'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'del', 'space'])
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    last = ''
    last_left = ''
    count = 0
    count_left = 0
    curr_sign = ''
    curr_sign_left = ''
    word = ''
    word_left = ''
    association = ''
    association_left = ''

    while cap.isOpened():  # while webcam is open, keep the loop
        ret, frame = cap.read()  # read the cap
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(frame, results)

        keypoints = extract_keypoints(results)
        keypoints_left = extract_keypoints_left(results)
        res = model.predict(keypoints.reshape(1, 63, 1))
        res_left = model.predict(keypoints_left.reshape(1, 63, 1))

        sign = alphabets[np.argmax(res)]
        if np.array_equal(keypoints, np.zeros(63)):
            sign = ' '
        if sign == last:
            count = count + 1
            if count == 10:
                print(sign)
                if sign == ' ' and word != '':
                    old_word = word
                    word = correct(word)
                    word += ' '
                    curr_sign += word
                    word = ''
                    association = ''
                if cv2.waitKey(10) & 0xFF == 127 and sign == ' ':
                    words = curr_sign.split()
                    if len(words) > 0:
                        words.pop()  # Remove the last word
                        new_sentence = " ".join(words)
                        curr_sign = new_sentence.join(old_word)
                if sign != ' ':
                    word += sign
                if len(word) >= 3:
                    association = associat(word)
        if cv2.waitKey(10) & 0xFF == 13:
            curr_sign += association
            curr_sign += ' '
            word = ''
            association = ''
        if sign != last:
            count = 1
            last = sign

        sign_left = alphabets[np.argmax(res_left)]
        if np.array_equal(keypoints_left, np.zeros(63)):
            sign_left = ' '
        if sign_left == last_left:
            count_left = count_left + 1
            if count_left == 10:
                print(sign_left)
                if sign_left == ' ' and word_left != '':
                    word_left = correct(word_left)
                    curr_sign_left += word_left
                    curr_sign_left += ' '
                    word_left = ''
                    association_left = ''
                if sign_left != ' ':
                    word_left += sign_left
                if len(word_left) >= 3:
                    association_left = associat(word_left)
        if cv2.waitKey(10) & 0xFF == 13:
            curr_sign_left += association_left
            curr_sign_left += ' '
            word_left = ''
            association_left = ''
        if sign_left != last_left:
            count_left = 1
            last_left = sign_left

        cv2.putText(frame, curr_sign_left + word_left, (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4,
                    cv2.LINE_AA)
        cv2.putText(frame, curr_sign + word, (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, association_left, (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, association, (120, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', frame)  # show the frame
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
