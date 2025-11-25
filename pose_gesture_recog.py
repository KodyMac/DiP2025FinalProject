import cv2
import numpy as np
import mediapipe as mp 
from collections import defaultdict
import time
import json
from pathlib import Path
import csv
from datetime import datetime


class PoseEstimator:
    #main pose estimation using mediapipe
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  #for better use with static images
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )

    def detect_pose(self,image):
        #detect pose in a single image
        rgb = cv2.cvtColor(image, cv2.COLRO_BGR2RGB)
        results = self.pose.process(rgb)
        return results
    
    def draw_pose(self, image, results):
        #draw skeleton on image
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return image
    
    def close(self):
        self.pose.close()


class GestureClassifier:
    
    GESTURE_MAP = {
        'left_hand_raised': 'hand_raise',
        'right_hand_raised': 'hand_raise',
        'both_hands_raised': 'hand_raise',
        'squatting': 'squat',
        'arms_crossed': 'arms_crossed',
        't_pose': 't_pose',
        'standing': 'standing',
        'no_person': 'no_detection'
    }

    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def classify(self, landmarks):
        #determine class of gesture using landmarks
        if not landmarks:
            return 'no_person', 0.0
        
        lw = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        rw = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        ls = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        rh = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

        #check hand raise
        left_raised = lw.y < ls.y - 0.1 #left hand lower than left shoulder
        right_raised = rw.y < rs.y -0.1

        if left_raised and right_raised:
            return 'both_hands_raised', 0.9
        elif left_raised:
            return 'left_hand_raised', 0.85
        elif right_raised:
            return 'right_hand_raised', 0.85
        
        #check for t pose
        left_horiz = abs(lw.y - ls.y) < 0.1 and lw.x < ls.x - 0.15
        right_horiz = abs(rw.y - rs.y) < 0.1 and rw.x > rs.x + 0.15
        if left_horiz and right_horiz:
            return 't_pose', 0.85
        
        #check for crossing arms
        mid_x = (ls.x + rs.x) / 2
        chest_y = (ls.y + rs.y) / 2
        crossed = (lw.x > mid_x and rw.x < mid_x and  #left wrist more right than center, right more left than center
                   abs(lw.y - chest_y) < 0.2 and abs(rw.y - chest_y) < 0.2)
        if crossed:
            return 'arms_crossed', 0.8
        
        #check for squat (hips lowered)
        avg_hip_y = (lh.y + rh.y) / 2
        avg_shoulder_y = (ls.y + rs.y) / 2
        torso_short = (avg_hip_y - avg_shoulder_y) < 0.3
        if torso_short:
            return 'squatting', 0.75
        
        return 'standing', 0.5









"""class HaGRIDEval:
    GESTURE_MAP = {   #HaGRID classes
        'call': 'Hand Gesture', 'dislike': 'Hand Gesture', 'fist': 'Hand Gesture',
        'four': 'Hand Gesture', 'like': 'Hand Gesture', 'mute': 'Hand Gesture',
        'ok': 'Hand Gesture', 'one': 'Hand Gesture', 'palm': 'Hand Gesture',
        'peace': 'Hand Gesture', 'peace_inverted': 'Hand Gesture',
        'rock': 'Hand Gesture', 'stop': 'Hand Gesture', 'stop_inverted': 'Hand Gesture',
        'three': 'Hand Gesture', 'three2': 'Hand Gesture', 'two_up': 'Hand Gesture',
        'two_up_inverted': 'Hand Gesture', 'no_gesture': 'No Gesture'
    }

    def __init__(self,dataset_path, annotations_path):
        self.dataset_path = Path(dataset_path)
        self.annotations_path = Path(annotations_path)
        self.recognizer = PoseGestureRecognizer()
        self.results = []

    def load_annotations(self, split='test'):
        #load HaGRID annotations
        annotations = {}
        ann_dir = self.annotations_path / split

        if not ann_dir.exists():
            print(f"Annotation directory {ann_dir} does not exist.")
            return annotations
        
        for json_file in ann_dir.glob('*.json'):
            gesture_class = json_file.stem
            with open(json_file, 'r') as f:
                data = json.load(f)
                for img_id, ann in data.items():
                    annotations[img_id] = {
                        'gesture': gesture_class,
                        'bboxes': ann.get('bboxes', []),
                        'labels': ann.get('labels', [])
                    }
        return annotations
    
    def evaluate(self, max_samples=100, visualize=False):
        #evaluate using HaGRID dataset
        print("=" * 20)
        print("HaGRID Evaluation")
        print("=" * 20)

        annotations = self.load_annotations('test')
        if not annotations:
            print("No annotations found. Exiting evaluation.")
            return
        
        #gropu by gesture class
        by_class = {}
        for img_id, ann in annotations.items():
            cls = ann['gesture']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append((img_id,ann))


        #sample from each class
        samples_per_class = max(1,max_samples // len(by_class))
        total_correct = 0
        total_samples = 0
        class_results = {}

        for gesture_class, items in by_class.items():
            samples = items[:samples_per_class]
            correct = 0

            for img_id, ann in samples:
                #find the image file
                img_path = self.dataset_path / gesture_class / f"{img_id}.jpg"
                if not img_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                self.recognizer.reset_state()
                processed, detected, conf = self.recognizer.process_frame(img.copy())

                #check if the detection is reasonable (person detected)
                person_detected = detected != "No Person Found"

                self.results.append({
                    'image': img_id,
                    'ground_truth': gesture_class,
                    'detected': detected,
                    'confidence': conf,
                    'person_detected': person_detected
                })

                if person_detected:
                    correct += 1
                total_samples += 1

                if visualize:
                    cv2.putText(processed, f"GT: {gesture_class}", (20, 160),cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2)
                    cv2.imshow('HaGrid Evaluation', processed)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break

            if samples:
                class_results[gesture_class] = {
                    'total': len(samples),
                    'detected': correct,
                    'accuracy': correct / len(samples) * 100
                }
                total_correct += correct

        if visualize:
            cv2.destroyAllWindows()

        #print results
        print(f"\nResults:")
        print("-"*20)
        print(f"{'Class': < 20} {'Detected': < 12} {'Total': < 10} {'Rate':<10}")
        print("-"*20)

        for cls, res in sorted(class_results.items()):
            print(f"{cls:<20} {res['detected']:<12} {res['total']:<10} {res['accuracy']:.1f}%")

        overall = total_correct / total_samples * 100 if total_samples else 0
        print("-"*20)
        print(f"{'OVERALL':<20} {total_correct:<12} {total_samples:<10} {overall:.1f}%")

        return class_results
    

class MPIIEval:
    #evaluate MPII human pose dataset"""