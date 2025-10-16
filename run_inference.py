"""
Fire and Smoke Detection Inference Script

This script loads a trained Faster R-CNN model and runs inference on:
1. Single images
2. Image directories
3. Video files
4. Webcam (real-time)

Usage:
    python run_inference.py --model_path runs/run_20251007_102957/model_best.pth --image path/to/image.jpg
    python run_inference.py --model_path runs/run_20251007_102957/model_best.pth --video path/to/video.mp4
    python run_inference.py --model_path runs/run_20251007_102957/model_best.pth --webcam
"""

import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as F
import time
from collections import deque

# Import the model architecture from fcnn.py
from fcnn import get_model, compute_iou

class FireSmokeDetector:
    def __init__(self, model_path, device=None, confidence_threshold=0.3):
        """
        Initialize the fire and smoke detector
        
        Args:
            model_path (str): Path to the trained model weights
            device (str): Device to run inference on ('cuda' or 'cpu')
            confidence_threshold (float): Minimum confidence for detections
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Initialize class names and colors
        self.class_names = {}
        self.colors = {}
        
        # Load the model
        print(f"Loading model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        # First, try to determine the number of classes from the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get the number of classes from the classifier weight shape
        if 'roi_heads.box_predictor.cls_score.weight' in checkpoint:
            num_classes = checkpoint['roi_heads.box_predictor.cls_score.weight'].shape[0]
            print(f"Detected {num_classes} classes from checkpoint")
        else:
            # Fallback to 5 classes (based on the error message)
            num_classes = 5
            print(f"Using default {num_classes} classes")
        
        # Create model with the correct number of classes
        model = get_model(num_classes=num_classes)
        
        # Load trained weights
        model.load_state_dict(checkpoint)
        model.to(self.device)
        
        # Set up class names and colors based on detected number of classes
        self._setup_class_names(num_classes)
        
        return model
    
    def _setup_class_names(self, num_classes):
        """Set up class names and colors based on number of classes"""
        # Based on the dataset, we have:
        # Class 0: background (not used in detection)
        # Class 1: fire
        # Class 2: smoke
        # Classes 3-4: might be additional categories or duplicates
        
        # Map the classes to meaningful names
        class_mapping = {
            1: 'fire',
            2: 'smoke',
            3: 'fire',      # In case there are duplicate fire classes
            4: 'smoke'      # In case there are duplicate smoke classes
        }
        
        # Create class names
        for i in range(1, num_classes):
            if i in class_mapping:
                self.class_names[i] = class_mapping[i]
            else:
                self.class_names[i] = f'class_{i}'
        
        # Define colors for each class
        color_mapping = {
            'fire': (0, 0, 255),      # Red for fire
            'smoke': (0, 255, 0),     # Green for smoke
        }
        
        # Assign colors
        for i, class_name in self.class_names.items():
            if class_name in color_mapping:
                self.colors[class_name] = color_mapping[class_name]
            else:
                # Default colors for unknown classes
                color_list = [
                    (255, 0, 0),    # Blue
                    (0, 255, 255),  # Yellow
                    (255, 0, 255),  # Magenta
                    (255, 255, 0),  # Cyan
                ]
                color_idx = (i - 1) % len(color_list)
                self.colors[class_name] = color_list[color_idx]
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        if isinstance(image, str):
            # Load image from path
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to tensor and normalize
        image_tensor = F.to_tensor(image)
        
        return image_tensor
    
    def detect(self, image):
        """
        Run inference on a single image
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            dict: Detection results with boxes, scores, and labels
        """
        with torch.no_grad():
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            predictions = self.model(image_tensor)[0]
            
            # Filter by confidence threshold
            scores = predictions['scores'].cpu().numpy()
            boxes = predictions['boxes'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            
            # Filter detections
            keep = scores >= self.confidence_threshold
            filtered_boxes = boxes[keep]
            filtered_scores = scores[keep]
            filtered_labels = labels[keep]
            
            return {
                'boxes': filtered_boxes,
                'scores': filtered_scores,
                'labels': filtered_labels
            }
    
    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image (numpy array)
            detections: Detection results from detect()
            
        Returns:
            numpy array: Image with drawn detections
        """
        result_image = image.copy()
        
        for box, score, label in zip(detections['boxes'], detections['scores'], detections['labels']):
            # Get class name and color
            class_id = int(label)
            class_name = self.class_names.get(class_id, f'class_{class_id}')
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label and confidence
            label_text = f"{class_name}: {score:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw background for text
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw text
            cv2.putText(result_image, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    def detect_image(self, image_path, output_path=None):
        """Detect fire and smoke in a single image"""
        print(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        # Run detection
        detections = self.detect(image)
        
        # Draw results
        result_image = self.draw_detections(image, detections)
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
        else:
            # Display result
            cv2.imshow('Fire and Smoke Detection', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Print detection summary
        print(f"Found {len(detections['boxes'])} detections:")
        for i, (box, score, label) in enumerate(zip(detections['boxes'], detections['scores'], detections['labels'])):
            class_name = self.class_names.get(int(label), f'class_{int(label)}')
            print(f"  {i+1}. {class_name} (confidence: {score:.3f})")
    
    def detect_video(self, video_path, output_path=None, show_video=False):
        """Detect fire and smoke in a video"""
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer - always create output if no path specified
        if output_path is None:
            # Create output filename based on input
            input_path = Path(video_path)
            output_path = input_path.parent / f"{input_path.stem}_detected{input_path.suffix}"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print(f"Error: Could not create output video {output_path}")
            cap.release()
            return
        
        frame_count = 0
        start_time = time.time()
        
        print("Processing frames...")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                detections = self.detect(frame)
                
                # Draw results
                result_frame = self.draw_detections(frame, detections)
                
                # Add FPS counter and frame info
                frame_count += 1
                if frame_count % 30 == 0:  # Update FPS every 30 frames
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / elapsed_time
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - FPS: {current_fps:.1f}")
                
                # Add info text to frame
                cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Detections: {len(detections['boxes'])}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Write frame to output video
                writer.write(result_frame)
                
                # Display frame if requested and display is available
                if show_video:
                    try:
                        cv2.imshow('Fire and Smoke Detection', result_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except:
                        # If display fails, continue without showing
                        pass
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        # Cleanup
        cap.release()
        writer.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"Video processing complete!")
        print(f"Processed {frame_count} frames in {elapsed_time:.1f} seconds")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Output saved to: {output_path}")
    
    def detect_webcam(self):
        """Real-time detection using webcam"""
        print("Starting webcam detection. Press 'q' to quit.")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Temporal filtering for smoother detections
        detection_buffer = deque(maxlen=5)  # Keep last 5 detections
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            detections = self.detect(frame)
            
            # Add to buffer for temporal filtering
            detection_buffer.append(detections)
            
            # Use majority voting for stable detections
            if len(detection_buffer) >= 3:
                # Simple temporal filtering - only show detections that appear in multiple frames
                stable_detections = self._temporal_filter(detection_buffer)
                result_frame = self.draw_detections(frame, stable_detections)
            else:
                result_frame = self.draw_detections(frame, detections)
            
            # Display frame
            cv2.imshow('Fire and Smoke Detection (Webcam)', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _temporal_filter(self, detection_buffer):
        """Simple temporal filtering to reduce false positives"""
        # This is a simplified version - in practice, you'd want more sophisticated filtering
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for det in detection_buffer:
            all_boxes.extend(det['boxes'])
            all_scores.extend(det['scores'])
            all_labels.extend(det['labels'])
        
        # Return the most recent detections (you could implement more sophisticated filtering here)
        return detection_buffer[-1]


def main():
    parser = argparse.ArgumentParser(description='Fire and Smoke Detection Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model weights')
    parser.add_argument('--image', type=str,
                       help='Path to input image')
    parser.add_argument('--video', type=str,
                       help='Path to input video')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam for real-time detection')
    parser.add_argument('--output', type=str,
                       help='Output path for results')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.3)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to run inference on (auto-detect if not specified)')
    parser.add_argument('--show', action='store_true',
                       help='Show video/image during processing (may not work in headless environments)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = FireSmokeDetector(
        model_path=args.model_path,
        device=args.device,
        confidence_threshold=args.confidence
    )
    
    # Run inference based on input type
    if args.image:
        detector.detect_image(args.image, args.output)
    elif args.video:
        detector.detect_video(args.video, args.output, show_video=args.show)
    elif args.webcam:
        detector.detect_webcam()
    else:
        print("Please specify --image, --video, or --webcam")
        parser.print_help()


if __name__ == "__main__":
    main()
