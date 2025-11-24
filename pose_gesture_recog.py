import cv2
import numpy as np
import mediapipe as mp  #figure this out
from collections import deque
import time
import json
import os
from pathlib import Path
import csv
from datetime import datetime


class PoseGestureRecognizer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose


class HaGRIDEval:
    GESTURE_MAP = {   #HaGRID classes
        'call': 'Hand Gesture', 'dislike': 'Hand Gesture', 'fist': 'Hand Gesture',
        'four': 'Hand Gesture', 'like': 'Hand Gesture', 'mute': 'Hand Gesture',
        'ok': 'Hand Gesture', 'one': 'Hand Gesture', 'palm': 'Hand Gesture',
        'peace': 'Hand Gesture', 'peace_inverted': 'Hand Gesture',
        'rock': 'Hand Gesture', 'stop': 'Hand Gesture', 'stop_inverted': 'Hand Gesture',
        'three': 'Hand Gesture', 'three2': 'Hand Gesture', 'two_up': 'Hand Gesture',
        'two_up_inverted': 'Hand Gesture', 'no_gesture': 'No Gesture'
    }