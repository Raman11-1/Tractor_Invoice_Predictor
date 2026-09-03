"""
utils/preprocessing.py - Image Preprocessing Pipeline
"""

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import io

class DocumentPreprocessor:
    """Handles document image preprocessing"""
    
    def __init__(self, dpi=300, target_size=(1024, 1024)):
        self.dpi = dpi
        self.target_size = target_size
    
    def load_document(self, file_path):
        """Load PDF or image document"""
        file_path = str(file_path)
        
        if file_path.lower().endswith('.pdf'):
            images = convert_from_path(file_path, dpi=self.dpi)
            return [np.array(img) for img in images]
        else:
            img = Image.open(file_path)
            return [np.array(img)]
    
    def enhance_image(self, image):
        """Apply image enhancement techniques"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Adaptive thresholding for better text extraction
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def correct_skew(self, image):
        """Correct document skew/rotation"""
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if abs(angle) > 0.5:  # Only correct if skew > 0.5 degrees
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        
        return image
    
    def resize_image(self, image, target_size=None):
        """Resize image while maintaining aspect ratio"""
        if target_size is None:
            target_size = self.target_size
        
        h, w = image.shape[:2]
        aspect = w / h
        
        if aspect > 1:
            new_w = target_size[0]
            new_h = int(new_w / aspect)
        else:
            new_h = target_size[1]
            new_w = int(new_h * aspect)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Pad to target size
        padded = np.zeros(target_size, dtype=image.dtype)
        y_offset = (target_size[1] - new_h) // 2
        x_offset = (target_size[0] - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def preprocess(self, file_path):
        """Complete preprocessing pipeline"""
        # Load document
        images = self.load_document(file_path)
        
        processed_images = []
        for img in images:
            # Convert to RGB if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Enhance
            enhanced = self.enhance_image(img)
            
            # Correct skew
            corrected = self.correct_skew(enhanced)
            
            # Resize
            resized = self.resize_image(corrected)
            
            processed_images.append({
                'original': img,
                'processed': resized,
                'enhanced': enhanced
            })
        
        return processed_images
    
    def extract_regions(self, image, padding=10):
        """Extract text regions from image"""
        # Find contours
        contours, _ = cv2.findContours(
            255 - image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 20:  # Filter small regions
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)
                
                region = image[y1:y2, x1:x2]
                regions.append({
                    'bbox': (x1, y1, x2, y2),
                    'region': region
                })
        
        return regions