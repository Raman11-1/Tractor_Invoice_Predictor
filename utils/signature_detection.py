"""
utils/signature_detection.py - Detect signatures and stamps using computer vision
"""

import cv2
import numpy as np
from ultralytics import YOLO

class SignatureStampDetector:
    """Detect signatures and stamps in documents"""
    
    def __init__(self, model_path=None, use_yolo=True):
        self.use_yolo = use_yolo
        self.model = None
        
        if use_yolo and model_path:
            self.model = YOLO(model_path)
        elif use_yolo:
            # Use pretrained YOLOv8 for general object detection
            # In practice, you'd fine-tune this on signature/stamp data
            self.model = YOLO('yolov8n.pt')
    
    def detect_with_cv(self, image):
        """Detect signatures and stamps using traditional CV"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Detect signatures (usually handwritten, irregular)
        signatures = self._detect_signatures_cv(gray)
        
        # Detect stamps (usually circular, colored)
        stamps = self._detect_stamps_cv(image)
        
        return {
            'signatures': signatures,
            'stamps': stamps
        }
    
    def _detect_signatures_cv(self, gray):
        """Detect signatures using contour analysis"""
        # Apply binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        signatures = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (signatures are typically medium-sized)
            if 1000 < area < 50000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Signatures are typically wider than tall
                if 1.5 < aspect_ratio < 10:
                    # Calculate density (signature should have moderate density)
                    roi = binary[y:y+h, x:x+w]
                    density = np.sum(roi > 0) / (w * h)
                    
                    if 0.05 < density < 0.5:
                        signatures.append({
                            'bbox': [x, y, x+w, y+h],
                            'confidence': min(density * 2, 0.9),
                            'area': area
                        })
        
        # Return highest confidence signature
        if signatures:
            best = max(signatures, key=lambda x: x['confidence'])
            return [best]
        
        return []
    
    def _detect_stamps_cv(self, image):
        """Detect stamps using color and shape analysis"""
        # Convert to HSV for color detection
        if len(image.shape) == 2:
            return []
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Detect red/blue stamps (common colors)
        # Red range
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask_red1 | mask_red2
        
        # Blue range
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Combine masks
        mask = mask_red | mask_blue
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stamps = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Stamps are typically circular and medium-sized
            if 500 < area < 30000:
                # Fit circle to check circularity
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circle_area = np.pi * radius * radius
                circularity = area / circle_area if circle_area > 0 else 0
                
                if circularity > 0.5:  # Reasonably circular
                    x, y, w, h = cv2.boundingRect(contour)
                    stamps.append({
                        'bbox': [int(x), int(y), int(x+w), int(y+h)],
                        'confidence': min(circularity, 0.9),
                        'area': area
                    })
        
        # Return highest confidence stamp
        if stamps:
            best = max(stamps, key=lambda x: x['confidence'])
            return [best]
        
        return []
    
    def detect_with_yolo(self, image):
        """Detect using YOLO model (if trained on signatures/stamps)"""
        if self.model is None:
            return {'signatures': [], 'stamps': []}
        
        # Run inference
        results = self.model(image, conf=0.25)
        
        signatures = []
        stamps = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                
                # Classify as signature or stamp based on class
                # (This assumes model is trained with these classes)
                if cls == 0:  # Signature class
                    signatures.append({
                        'bbox': bbox,
                        'confidence': conf
                    })
                elif cls == 1:  # Stamp class
                    stamps.append({
                        'bbox': bbox,
                        'confidence': conf
                    })
        
        return {
            'signatures': signatures,
            'stamps': stamps
        }
    
    def detect(self, image):
        """Main detection method"""
        if self.use_yolo and self.model is not None:
            try:
                return self.detect_with_yolo(image)
            except:
                pass
        
        # Fallback to CV methods
        return self.detect_with_cv(image)
    
    def format_detection(self, detections):
        """Format detection results for output"""
        signatures = detections.get('signatures', [])
        stamps = detections.get('stamps', [])
        
        signature_result = {
            'present': len(signatures) > 0,
            'bbox': signatures[0]['bbox'] if signatures else [0, 0, 0, 0],
            'confidence': signatures[0]['confidence'] if signatures else 0.0
        }
        
        stamp_result = {
            'present': len(stamps) > 0,
            'bbox': stamps[0]['bbox'] if stamps else [0, 0, 0, 0],
            'confidence': stamps[0]['confidence'] if stamps else 0.0
        }
        
        return signature_result, stamp_result