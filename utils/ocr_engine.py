"""
utils/ocr_engine.py - Multi-OCR Engine with Ensemble
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
import easyocr
import pytesseract
from collections import Counter
import re

class MultiOCREngine:
    """Ensemble of multiple OCR engines for robust text extraction"""
    
    def __init__(self, languages=['en', 'hi'], use_gpu=False):
        self.languages = languages
        self.use_gpu = use_gpu
        
        # Initialize OCR engines
        self.paddle_ocr = PaddleOCR(
            lang='en',
            use_angle_cls=True,
            use_gpu=use_gpu,
            show_log=False
        )
        
        self.easy_ocr = easyocr.Reader(
            languages,
            gpu=use_gpu
        )
    
    def extract_with_paddle(self, image):
        """Extract text using PaddleOCR"""
        result = self.paddle_ocr.ocr(image, cls=True)
        
        extracted_text = []
        for line in result[0] if result[0] else []:
            bbox = line[0]
            text = line[1][0]
            conf = line[1][1]
            
            extracted_text.append({
                'text': text,
                'confidence': conf,
                'bbox': bbox,
                'engine': 'paddle'
            })
        
        return extracted_text
    
    def extract_with_easy(self, image):
        """Extract text using EasyOCR"""
        result = self.easy_ocr.readtext(image)
        
        extracted_text = []
        for bbox, text, conf in result:
            extracted_text.append({
                'text': text,
                'confidence': conf,
                'bbox': bbox,
                'engine': 'easy'
            })
        
        return extracted_text
    
    def extract_with_tesseract(self, image):
        """Extract text using Tesseract"""
        # Get detailed data
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            lang='eng+hin',
            config='--oem 3 --psm 6'
        )
        
        extracted_text = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:
                text = data['text'][i].strip()
                if text:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                    
                    extracted_text.append({
                        'text': text,
                        'confidence': int(data['conf'][i]) / 100.0,
                        'bbox': bbox,
                        'engine': 'tesseract'
                    })
        
        return extracted_text
    
    def ensemble_extract(self, image):
        """Extract text using multiple OCR engines and combine results"""
        results = []
        
        # Try PaddleOCR
        try:
            paddle_results = self.extract_with_paddle(image)
            results.extend(paddle_results)
        except Exception as e:
            print(f"PaddleOCR failed: {e}")
        
        # Try EasyOCR
        try:
            easy_results = self.extract_with_easy(image)
            results.extend(easy_results)
        except Exception as e:
            print(f"EasyOCR failed: {e}")
        
        # Try Tesseract
        try:
            tess_results = self.extract_with_tesseract(image)
            results.extend(tess_results)
        except Exception as e:
            print(f"Tesseract failed: {e}")
        
        return results
    
    def get_full_text(self, ocr_results):
        """Combine OCR results into full text"""
        texts = [r['text'] for r in ocr_results]
        return ' '.join(texts)
    
    def extract_numbers(self, text):
        """Extract all numbers from text"""
        # Match numbers with optional decimal points
        numbers = re.findall(r'\d+\.?\d*', text)
        return [float(n) if '.' in n else int(n) for n in numbers]
    
    def extract_text_by_pattern(self, ocr_results, pattern):
        """Extract text matching a specific pattern"""
        matches = []
        for result in ocr_results:
            text = result['text']
            if re.search(pattern, text, re.IGNORECASE):
                matches.append({
                    'text': text,
                    'confidence': result['confidence'],
                    'bbox': result['bbox']
                })
        return matches
    
    def consensus_text(self, texts, weights=None):
        """Get consensus text from multiple OCR outputs"""
        if not texts:
            return ""
        
        if weights is None:
            weights = [1.0] * len(texts)
        
        # Normalize and clean texts
        cleaned_texts = []
        for text in texts:
            # Remove extra spaces, convert to lowercase for comparison
            cleaned = ' '.join(text.split()).lower()
            cleaned_texts.append(cleaned)
        
        # Find most common text
        if len(set(cleaned_texts)) == 1:
            return texts[0]
        
        # Weight-based voting
        weighted_counts = {}
        for text, weight in zip(texts, weights):
            weighted_counts[text] = weighted_counts.get(text, 0) + weight
        
        best_text = max(weighted_counts.items(), key=lambda x: x[1])[0]
        return best_text
    
    def extract_structured_data(self, image):
        """Extract structured data from invoice"""
        ocr_results = self.ensemble_extract(image)
        full_text = self.get_full_text(ocr_results)
        
        return {
            'ocr_results': ocr_results,
            'full_text': full_text,
            'numbers': self.extract_numbers(full_text)
        }