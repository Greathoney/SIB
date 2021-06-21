# File name: livemodel_mlp.py
# Platform: Python 3.8.8 on Ubuntu Linux 18.04
# Required Package(s): cv2, mediapipe, numpy, tensorflow, pynput
# Date: 2021.06.19
# Name: Dohun Kim, DaeHeon Yoon


################################# import packages #################################

import os, sys
sys.path.append(os.pardir)

import time
from multiprocessing import Process, Queue, Value
from ctypes import c_bool

import numpy as np
from tensorflow.keras.models import load_model

import cv2
import mediapipe as mp

from pynput import keyboard
import termios


############################# shared global variables #############################

queue = Queue()
flag_stop = Value(c_bool, False)


########################### key prediction - tensorflow ###########################

def model_thread(model_path):

    # load pretrained tensorflow model
    model = load_model(model_path)

    model.summary()
    
    while not flag_stop.value:
        if queue.empty():
            continue

        hand_np = queue.get()
        pred = model.predict(hand_np)
        # 만일 "None"레이블의 민감도가 너무 낮으면 올려주고, 높으면 낮춰주는 로직 후처리1
        # sensitivity <- 우리가 지정, 자동으로 지정될 수 있도록?!

        # 0일때 가중치가 0.6이상이면 1로 바꾸고, 1인 친구는 0.4 이하면 0으로 바꾸는 후처리2
        print(np.argmax(pred))

    print('model_thread() terminated.')


############################ hand landmark - mediapipe ############################

def hand_to_numpy(hand_data):
    if hand_data.multi_hand_landmarks:
        result = []
        for landmark in hand_data.multi_hand_landmarks[0].landmark:
            result += [landmark.x, landmark.y, landmark.z]
        return np.array([result])
    return None


# def hand_preprocessing(hand_np, cut_outlier=False, gamma_smoothing=False, local_minmax=False):
#     if cut_outlier:
#         # 라이브 버전의 cut_outlier를 만들어야 함
#         # cout_outlier 전체 평균의 표준편차 <- 잘라내는 걸 확인
#         # 성능에 부담??? <- 여의치 않으면 빼야할 수도?
#         pass

#     if gamma_smoothing:
#         # hand_np = gamma_smoothing(hand_np)
#         # 라이브 포맷으로 바꿔주어야 함
#         # 기존에는 이전 데이터 하나 가져오기
#         # 이번에는 이전 프레임 데이터를 저장 -> 전처리
#         hand_np = smoothing.process(hand_np)

#     if local_minmax:    # preprocessing - Local MinMaxScaler
#         global local_min, local_max

#         reduce_pole = 0.00001
#         local_min = np.array(list(map(lambda x,y:min(x, y) * (1 + reduce_pole), local_min, hand_np)))
#         local_max = np.array(list(map(lambda x,y:max(x, y) * (1 - reduce_pole), local_max, hand_np)))

#         hand_np = map(lambda x,y,z:(x-y)/(z-y), hand_np, local_min, local_max)
    
#     return hand_np

class CutOutlier:
    pass

class GammaSmoothing:
    def __init__(self, gamma=0.4):
        self.gamma = gamma
        self.prev  = None

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = x
            return x
    
        result = (1-self.gamma)*self.prev + self.gamma*x
        self.prev = result
        return result     

class LocalMinMax:
    def __init__(self, decay_rate=1e-5):
        self.decay_rate = decay_rate
        self.local_min = None
        self.local_max = None
    
    def process(self, x: np.ndarray) -> np.ndarray:
        if self.local_min is None:
            self.local_min = x
        if self.local_max is None:
            self.local_max = x
        
        self.local_min = np.where(x < self.local_min, x, self.local_min) * (1 + self.decay_rate)
        self.local_max = np.where(x > self.local_max, x, self.local_max) * (1 - self.decay_rate)

        return (x - self.local_min) / (self.local_max - self.local_min)


class Preprocessing:
    def __init__(self, cut_outlier=False, gamma_smoothing=False, local_minmax=False):
        self.tasks = []
        if cut_outlier:
            self.tasks += [CutOutlier()]  
        if gamma_smoothing:
            self.tasks += [GammaSmoothing()]
        if local_minmax:
            self.tasks += [LocalMinMax()]

    def process(self, x: np.ndarray) -> np.ndarray:
        for task in self.tasks:
            x = task.process(x)
        return x



FPS     = 30
TIMEOUT = 1 / FPS

def hand_thread(flip=False, debug=False):

    # mediapipe hands shortcut
    mp_hands = mp.solutions.hands

    # webcam input
    if not debug:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('cv2.VideoCapture open failed.')
            return
    
    # hands data preprocessor
    preprocessor = Preprocessing(gamma_smoothing=True,
                                 local_minmax=True)
    
    # create mediapipe hands module
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    
        # while-loop with fixed frame rate(FPS)
        old_timestamp = time.time()
        while not flag_stop.value:
            if (time.time() - old_timestamp) <= TIMEOUT:
                continue
            # print('FPS: %.3f' % (1/(time.time() - old_timestamp)))

            old_timestamp = time.time()

            # get image from file / webcam
            if debug:
                image = cv2.imread('../examples/mediapipe/test_image_1.jpg')
            else:            
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if flip:
                image = cv2.flip(image, 1)
            
            # get hand landmark data from image
            image.flags.writeable = False
            hand_data = hands.process(image)

            hand_np = hand_to_numpy(hand_data)

            if hand_np is not None:
                hand_np = preprocessor.process(hand_np)
                queue.put(hand_np)
    
    cap.release()
    print('hand_thread() terminated.')



####################################### main ######################################

def on_press(key):
    if key == keyboard.Key.esc:
        flag_stop.value = True
        return False

def flush_input():
    termios.tcflush(sys.stdin, termios.TCIOFLUSH)


if __name__ == '__main__':

    model_path = 'saved_model/model_mlp_space.h5'
    
    model = Process(target=model_thread, kwargs={'model_path': model_path})
    hand  = Process(target=hand_thread,  kwargs={'debug': False})

    model.start()
    hand.start()

    with keyboard.Listener(
            on_press=on_press) as listener:
        listener.join()
    
    flush_input()
    
    model.join()
    hand.join()