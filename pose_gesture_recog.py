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
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        'thumbs_up': 'thumbs_up',
        'peace': 'peace',
        'fist': 'fist',
        'leaning_left': 'leaning_left',
        'leaning_right': 'leaning_right',
        'pointing': 'pointing',
        'neutral': 'neutral',
        'no_person': 'no_person'
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
        lthumb = landmarks[self.mp_pose.PoseLandmark.LEFT_THUMB]
        rthumb = landmarks[self.mp_pose.PoseLandmark.RIGHT_THUMB]
        lindex = landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX]
        rindex = landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX]

        left_thumb_up = (lthumb.y < lw.y - 0.05 and #thumb above wrist
                        lw.y < ls.y and #hand raised
                        lthumb.visibility > 0.5)  #thumb visible
        
        right_thumb_up = (rthumb.y < rw.y - 0.05 and #thumb above wrist
                        rw.y < rs.y and #hand raised
                        rthumb.visibility > 0.5)  #thumb visible 
        if left_thumb_up or right_thumb_up:
            return 'thumbs_up', 0.85

        left_peace = (lindex.y < lw.y - 0.05 and #index above wrist
                      lw.y < ls.y and #hand raised
                      lindex.visibility > 0.5)  #index visible  
        right_peace = (rindex.y < rw.y - 0.05 and #index above wrist
                      rw.y < rs.y and #hand raised
                      rindex.visibility > 0.5)  #index visible
        if left_peace or right_peace:
            return 'peace', 0.80

        lfist = (lw.y < ls.y and #hand raised
                 abs(lthumb.y - lw.y) < 0.05 and #thumb not extended
                 abs(lindex.y - lw.y) < 0.05) #index not extended
        rfist = (rw.y < rs.y and #hand raised
                 abs(rthumb.y - rw.y) < 0.05 and #thumb not extended
                 abs(rindex.y - rw.y) < 0.05) #index not extended
        if lfist or rfist:
            return 'fist', 0.75
        
        shoulder_slope = abs(ls.y - rs.y)
        hip_slope = abs(lh.y - rh.y)
        #if either shoulder or hip is tilted, person is leaning
        if shoulder_slope > 0.08 or hip_slope > 0.08:
            if ls.y < rs.y: #left shoulder higher, so leaning right
                return 'leaning_right', 0.75
            else: #right shoulder higher, so leaning left
                return 'leaning_left', 0.75
            
        body_center_x = (ls.x + rs.x) / 2
        #left arm extended to left
        left_pointing = (lw.x < body_center_x - 0.2 and #arem extended left
                         lindex.visibility > 0.5 and #index visible
                         abs(lw.y - ls.y) < 0.15) #arm about horizontal
        right_pointing = (rw.x > body_center_x + 0.2 and #arm extended right
                          rindex.visibility > 0.5 and
                          abs(rw.y - rs.y) < 0.15)
        if left_pointing or right_pointing:
            return 'pointing', 0.70
        
        return 'neutral', 0.5
    
        """#check hand raise
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
        
        return 'standing', 0.5"""


#creates custom dataset (annotated dataset from webcam)
class CustomDatasetCreator:
    GESTURES = ['thumbs_up', 'peace', 'fist', 'leaning', 'pointing']

    def __init__(self, output_dir='custom_dataset'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        for gesture in self.GESTURES:
            (self.output_dir / gesture).mkdir(exist_ok=True)

        self.annotations = []
        print(f"\n Dataset directory created: {self.output_dir}")

    def record_gesture(self, gesture_name, num_samp = 20):
        if gesture_name not in self.GESTURES:
            print(f"\nUnknown gesture. Choose from: {self.GESTURES}")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot access webcam")
            return
        
        save_dir = self.output_dir / gesture_name
        samples_saved = 0

        print(f"\n Recording '{gesture_name}'")
        print(f"  Target: {num_samp} samples")
        print(f"   Controls: SPACE=caputure, Q=quit, S=skip")

        while cap.isOpened() and samples_saved < num_samp:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            h,w= display.shape[:2]

            #overlay info
            overlay = display.copy()
            cv2.rectangle(overlay, (10,10), (w-10, 120), (0,0,0), -1)
            display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)

            cv2.putText(display, f"Gesture: {gesture_name.upper()}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)
            cv2.putText(display, f"Captured: {samples_saved}/{num_samp}", (20,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)
            cv2.putText(display, "SPACE=Capture Q=quit S=skip", (20,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
            
            cv2.imshow('Dataset Recording', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '): #space to caputure
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f') #get capture time (year,month, day, hour, minute, second)
                filename = f"{gesture_name}_{timestamp}.jpg"
                filepath = save_dir / filename

                cv2.imwrite(str(filepath),frame)
                self.annotations.append({
                    'filename': filename,
                    'gesture': gesture_name,
                    'path': str(filepath.relative_to(self.output_dir)),
                    'timestamp': timestamp
                })
                samples_saved += 1
                print(f" Captured {samples_saved}/{num_samp}: {filename}")
                
                cv2.waitKey(300)

            elif key == ord('q'): #quit
                break
            elif key == ord('s'): #skip current gesture
                print(f" Skip '{gesture_name}' ({samples_saved} samples saved)")
                break

        cap.release()
        cv2.destroyAllWindows()

        self._save_annotations()
        print(f" Saved {samples_saved} samples for '{gesture_name}'")

    def record_all_gestures(self, samples_per_gesture=20):
        print("\n" + "="*20)
        print("Custom Dataset Recording")
        print("="*20)
        print(f"Gestures to record: {', '.join(self.GESTURES)}")
        print(f"Samples per gesture: {samples_per_gesture}")

        for gesture in self.GESTURES:
            response = input(f"\nRecord '{gesture}'? (y/n/q): ").lower()

            if response == 'q':
                break
            elif response == 'y':
                self.record_gesture(gesture, samples_per_gesture)

        print("\n" + "="*20)
        print(f"Dataset recording finished")
        print(f"Total samples recorded: {len(self.annotations)}")
        print(f" Saved to: {self.output_dir.absolute()}")
        print("="*20)

    def _save_annotations(self):
        ann_file = self.output_dir / 'annotations.json'
        with open(ann_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
            

    #dataset evals
class CustomDatasetEval:
    def __init__(self, dataset_path= 'custom_dataset'):
        self.dataset_path = Path(dataset_path)
        self.estimator =PoseEstimator()
        self.classifier = GestureClassifier()
        self.results = []

    def evaluate(self, visualize = False, save_vis =False):
        ann_file = self.dataset_path / 'annotations.json'
        if not ann_file.exists():
            print(f" No annotations found at {ann_file}.\n Please create custom dataset first.")
            return None
            
        with open(ann_file, 'r') as f:
            annotations = json.load(f)

        if not annotations:
            print("Empty annotations file")
            return None
            
        print("\n" + "="*20)
        print("Custom Dataset Evaluation")
        print("="*20)
        print(f"Dataset: {self.dataset_path}")
        print(f"Total samples: {len(annotations)}")

        #organize
        by_class = defaultdict(list)
        for ann in annotations:
            by_class[ann['gesture']].append(ann)

        print(f"Classes: {list(by_class.keys())}")
        print("="*20)

        #evaluate each image
        class_results = defaultdict(lambda: {'correct':0, 'total':0, 'detections': []})
            
        viz_dir = None
        if save_vis:
            viz_dir = self.dataset_path / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
            print(f"\n Saving visualizations to : {viz_dir}")

        for i, ann in enumerate(annotations):
            img_path = self.dataset_path / ann['path']
            if not img_path.exists():
                print(f"Missing {img_path}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            #detect the pose
            pose_results = self.estimator.detect_pose(img)

            #classify gesture
            landmarks = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None
            detected_gest, confidence = self.classifier.classify(landmarks)

            #mpa the detected to a dataset label
            detected_label = self.classifier.GESTURE_MAP.get(detected_gest, 'unknown')
            gt_label = ann['gesture']

            #record the result
            correct = (detected_label == gt_label)
            class_results[gt_label]['total'] += 1
            if correct:
                class_results[gt_label]['correct'] += 1

            class_results[gt_label]['detections'].append({
                'file': ann['filename'],
                'detected': detected_label,
                'confidence': confidence,
                'correct': correct
            })

            #visualize
            if visualize or save_vis:
                vis_img = img.copy()
                vis_img = self.estimator.draw_pose(vis_img, pose_results)

                if i < 2:
                    report_dir = self.dataset_path / 'report_samples'
                    report_dir.mkdir(exist_ok=True)
                    report_path = report_dir / f"report_sample_{i+1}_{gt_label}_{detected_label}.jpg"
                    cv2.imwrite(str(report_path), vis_img)
                    print(f" Saved report sample to: {report_path}")

                #add labels
                color = (0,255,0) if correct else (0,0,255)
                cv2.putText(vis_img, f"GT: {gt_label}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
                cv2.putText(vis_img, f"Pred: {detected_label} ({confidence:.2f})", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
                cv2.putText(vis_img, "CORRECT" if correct else "WRONG", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                if save_vis:
                    viz_path = viz_dir / f"viz_{ann['filename']}"
                    cv2.imwrite(str(viz_path), vis_img)

                if visualize:
                    cv2.imshow('Evaluation', vis_img)
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        break
                    
            #tell progress
            if (i +1) % 20 == 0:
                print(f" Processed {i+1}/{len(annotations)} images...")

        if visualize:
            cv2.destroyAllWindows()
        self.estimator.close()
        #print results
        self.print_results(class_results)
        return class_results


    def print_results(self, class_results):
        print("\n" + "="*20)
        print("RESULTS")
        print("="*20)
        print(f"{'Gesture':<20} {'Correct': <10} {'Total':<10} {'Accuracy':<10}")
        print("="*20)

        total_cor = 0
        total_samp = 0

        for gesture in sorted(class_results.keys()):
            res = class_results[gesture]
            acc = (res['correct'] / res['total'] * 100) if res['total'] > 0 else 0
            print(f"{gesture:<20} {res['correct']:<10} {res['total']:<10} {acc:.1f}%")
            total_cor += res['correct']
            total_samp = res['total']

        overall_acc = (total_cor / total_samp * 100) if total_samp > 0 else 0
        print("-"*20)
        print(f"{'OVERALL':<20} {total_cor:<10} {total_samp:<10} {overall_acc:.1f}%")
        print("-"*20)


#HaGRID is hand gestures
class HaGRIDEval:

    def __init__(self, dataset_path, annotations_path):
        self.dataset_path = Path(dataset_path)
        self.annotations_path = Path(annotations_path)
        self.estimator = PoseEstimator()
    
    def evaluate(self, max_samples=100, visualize=False):
        #evaluate using HaGRID dataset
        print("=" * 20)
        print("HaGRID Evaluation")
        print("=" * 20)
        print(f"Dataset: {self.dataset_path}")
        print(f"Max samples per class: {max_samples}")

        #find gesture class dirs
        gesture_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]

        #filter non-gesture dirs
        valid_gesture_dirs = []
        for d in gesture_dirs:
            jpg_files = list(d.glob('*.jpg'))
            if jpg_files:
                valid_gesture_dirs.append(d)
            else:
                print(f" Skipping empty gesture dir: {d.name}")

        gesture_dirs = valid_gesture_dirs

        if not gesture_dirs:
            print(f"No gesture directories found in {self.dataset_path}")
            return None
        
        print(f"Found {len(gesture_dirs)} gesture classes")
        print("="*20)

        results = defaultdict(lambda: {'detected': 0, 'total': 0})
        
        total_proc = 0
        for gesture_idx, gesture_dir in enumerate(gesture_dirs):
            gesture_name = gesture_dir.name
            all_image_files = list(gesture_dir.glob('*.jpg'))
            image_files = all_image_files[:max_samples]

            if not image_files:
                print(f"No images found in {gesture_dir}/, skipping...")
                continue
            
            print(f"\n[{gesture_idx+1}/{len(gesture_dirs)}] Processing '{gesture_name}': {len(image_files)} images (from {len(all_image_files)} total)")

            #process each image
            for img_idx, img_path in enumerate(image_files):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                pose_results = self.estimator.detect_pose(img)
                person_det = pose_results.pose_landmarks is not None

                results[gesture_name]['total'] += 1
                if person_det:
                    results[gesture_name]['detected'] += 1
                
                total_proc += 1

                #visualize
                if visualize:
                    vis_img = img.copy()
                    vis_img = self.estimator.draw_pose(vis_img, pose_results)

                    status = "DETECTED" if person_det else "NOT DETECTED"
                    color = (0,255,0) if person_det else (0,0,255)

                    if img_idx == 0: #save some samples for report
                        report_dir = self.dataset_path.parent / 'hagrid_report_samples'
                        report_dir.mkdir(exist_ok=True)

                        original_path = report_dir / f"hagrid_{gesture_name}_original.jpg"
                        cv2.imwrite(str(original_path), img)
                        print(f" Saved original sample to: {original_path}")
                        report_path = report_dir / f"hagrid_{gesture_name}_{status.replace(' ', '_')}.jpg"
                        cv2.imwrite(str(report_path), vis_img)
                        print(f" Saved sample visualization to: {report_path}")                    

                    cv2.putText(vis_img, f"{gesture_name}: {status}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(vis_img, f"Image {img_idx+1}/{len(image_files)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)
                    cv2.imshow('HaGRID Evaluation', vis_img)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
            
            #show summary
            print(f"    {gesture_name}: {results[gesture_name]['detected']}/{results[gesture_name]['total']} detected ({results[gesture_name]['detected']/results[gesture_name]['total']*100:.1f}%)")

        print(f"\nTotal images processed: {total_proc}")

            #clean up
        if visualize:
            cv2.destroyAllWindows()

        self.estimator.close()

        self.print_results(results)

        return results
    
    def print_results(self, results):
        print("\n" + "="*20)
        print("RESULTS: Detection Rate")
        print("="*20)
        print(f"{'Class':<20} {'Detected':<12} {'Total':<10} {'Rate':<10}")
        print("-"*20)

        total_det = 0
        total_samp = 0

        for gesture in sorted(results.keys()):
            res = results[gesture]
            rate = (res['detected'] / res['total'] * 100) if res['total'] > 0 else 0
            print(f"{gesture:<20} {res['detected']:<12} {res['total']:<10} {rate:.1f}%f")
            total_det += res['detected']
            total_samp += res['total']

        overall_rate = (total_det / total_samp * 100) if total_samp > 0 else 0
        print("-"*20)
        print(f"{'OVERALL':<20} {total_det:<12} {total_samp:<10} {overall_rate:.1f}%")
        print("="*20)

class MPIIEval:
    def __init__(self,dataset_path):
        self.dataset_path = Path(dataset_path)
        self.estimator = PoseEstimator()

    def evaluate(self, max_samples=100, visualize=False):
        print("\n" + "="*20)
        print("MPII Dataset Evaluation")
        print("="*20)
        print(f"Dataset: {self.dataset_path}")
        print(f"Max samples: {max_samples}")
        print("="*20)

        #find images
        image_files = list(self.dataset_path.glob('*.jpg'))[:max_samples]

        if not image_files:
            print(f"No images found in {self.dataset_path}")
            return None
        
        print(f"Found {len(image_files)} images")

        det_count = 0
        tot_count = 0

        for i, img_path in enumerate(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            pose_results = self.estimator.detect_pose(img)
            person_det = pose_results.pose_landmarks is not None

            total_count += 1
            if person_det:
                det_count += 1

            if visualize:
                vis_img = img.copy()
                vis_img = self.estimator.draw_pose(vis_img, pose_results)
                status = "DETECTED" if person_det else "NOT DETECTED"
                color = (0,255,0) if person_det else (0,0,255)
                cv2.putText(vis_img, status, (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(vis_img, f"Image {i+1}/{len(image_files)}", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)
                cv2.imshow('MPII Evaluation', vis_img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

            if(i+1) % 20 == 0:
                print(f"    Processed {i+1}/{len(image_files)} images...")
        
        if visualize:
            cv2.destroyAllWindows()

        self.estimator.close()

        #print results
        detection_rate = (det_count / tot_count * 100) if total_count > 0 else 0
        print("\n" + "="*20)
        print("Results")
        print("="*20)
        print(f"Images processed: {tot_count}")
        print(f"Poses detected: {det_count}")
        print(f"Detection rate: {detection_rate:.1f}%")
        print("="*20)

        return {'detected': det_count, 'total': tot_count, 'rate': detection_rate}
    

class ResultsExporter:    
    @staticmethod
    def export_report(all_results, output_file='evaluation_reports.txt'):
        with open(output_file, 'w') as f:
            f.write("="*20 + "\n")
            f.write("2D Pose Gesture Recognition Evaluation Report\n")
            f.write("Evaluation Report\n")
            f.write("="*20 + "\n\n")
                #timestamp
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                #wirte results for each dataset
            for dataset_name, results in all_results.items():
                f.write(f"\n{dataset_name}\n")
                f.write("-"*20 + "\n")

                if results:
                    if isinstance(results, dict):
                        for key, value in results.items():
                            if isinstance(value, dict):
                                f.write(f"  {key}:\n")
                                for k, v in value.items():
                                    f.write(f"    {k}: {v}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                else:
                    f.write(" No results available.\n")
            f.write("\n" + "="*20 + "\n")
        print(f"\n Evaluation report saved to {output_file}") 
    @staticmethod
    def export_csv(results, output_file='evaluation_results.csv'):
        rows = []
        for dataset, data in results.items():
            if isinstance(data,dict):
                for key, value in data.items():
                    if isinstance(value,dict):
                        for k, v in value.items():
                            rows.append([dataset, key, k, v])
                    else:
                        rows.append([dataset,key, '', value])
        #write csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Dataset', 'Category', 'Metric', 'Value'])
            writer.writerows(rows)
        print(f" CSV saved to: {output_file}")           


def print_menu():
    print("\n" + "="*20)
    print("\nDataset Creation:")
    print("  1. Record custom dataset")
    print("\nDataset Evaluation")
    print("  2. Evaluate custom dataset")
    print("  3. Evaluate HaGRID dataset")
    print("  4. Evaluate MPII dataset")
    print("\nResults:")
    print("  5. Export evaluation report")
    print("  0. Exit")
    print("-"*20)


def main():
    all_results = {}

    print("DiP Final Project")

    while True:
        print_menu()
        choice = input("\nSelect option: ").strip()

        if choice == '1':
            output_dir = input("Output directory [custom_dataset]: ").strip() or 'custom_dataset'
            samples = input("Samples per gesture [20]: ").strip()
            samples = int(samples) if samples else 20

            creator = CustomDatasetCreator(output_dir)
            creator.record_all_gestures(samples_per_gesture=samples)

        elif choice == '2':
            dataset_path = input("Dataset path [custom_dataset]: ").strip() or 'custom_dataset'

            if not Path(dataset_path).exists():
                print(f"Directory not found: {dataset_path}")
                continue

            vis = input("Visualize during evaluation? (y/n) [n]: ").lower() == 'y'
            save_vis = input("Save visualizations? (y/n) [n]: ").lower() == 'y'

            evaluator = CustomDatasetEval(dataset_path)
            results = evaluator.evaluate(visualize = vis, save_vis=save_vis)

            if results:
                all_results['Custom Dataset'] = results

        elif choice == '3':
            default_paths = [
                'datasets/hagrid/images',
                'datasets\\hagrid\\images',
                '../datasets/hagrid/images',
                '..\\datasets\\hagrid\\images'
            ]

            found_path = None
            for p in default_paths:
                if Path(p).exists():
                    found_path = p
                    break

            if found_path:
                print(f"\n Found HaGRID dataset at: {found_path}")
                use_default = input("Use this path? (y/n) [y]: ").lower()
                if use_default == 'n':
                    dataset_path = input(" Enter HaGRID images path: ").strip()
                else:
                    dataset_path = found_path
            else:
                dataset_path = input("HaGRID images path [datasets/hagrid/images]: ").strip()
                dataset_path = dataset_path or 'datasets/hagrid/images'

            if not Path(dataset_path).exists(): #validate path
                print(f" Directory not found: {dataset_path}")
                print(f" Expected structure: {dataset_path}/call/, {dataset_path}/like/)")
                continue

            gesture_dirs = [d for d in Path(dataset_path).iterdir() if d.is_dir()]
            if not gesture_dirs:
                print(f"No gesture directories found in {dataset_path}")
                continue

            print(f" Found {len(gesture_dirs)} gesture classes: {[d.name for d in gesture_dirs[:5]]}{'...' if len(gesture_dirs) > 5 else ''}")

            ann_path = input("HaGRID annotations path [Enter to skip]: ").strip()

            vis = input("Visualize? (y/n) [n]: ").lower() == 'y'
            max_samp = input("Max samples per class [100]: ").strip()
            max_samp = int(max_samp) if max_samp else 100

            print(f"\n Starting HaGRID evaluation...")

            #run eval
            evaluator = HaGRIDEval(dataset_path, ann_path)
            results = evaluator.evaluate(max_samples=max_samp, visualize=vis)

            if results:
                all_results['HaGRID'] = results

        #elif choice == '4':   #to do later if time (MPII eval)
        #    break

        #make report
        elif choice == '5':
            if not all_results:
                print("No evaluation results to save.")
                continue

            #export in both forms
            ResultsExporter.export_report(all_results)
            ResultsExporter.export_csv(all_results)
            print("Reports exported.")
        
        elif choice == '0':
            print("\nGoodbye")
            break
        else:
            print("Invalid option.")


if __name__ == '__main__':
    main()        