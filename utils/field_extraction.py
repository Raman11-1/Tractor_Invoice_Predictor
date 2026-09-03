"""
utils/field_extraction.py - Extract specific fields from OCR results
"""

import re
import pandas as pd
from rapidfuzz import fuzz, process
import numpy as np

class FieldExtractor:
    """Extract specific fields from invoice documents"""
    
    def __init__(self, dealer_master_path=None, model_master_path=None):
        # Load master files if provided
        self.dealer_master = None
        self.model_master = None
        
        if dealer_master_path:
            self.dealer_master = pd.read_csv(dealer_master_path)
        
        if model_master_path:
            self.model_master = pd.read_csv(model_master_path)
    
    def extract_dealer_name(self, ocr_results, full_text):
        """Extract and match dealer name"""
        # Common dealer indicators
        dealer_keywords = [
            r'dealer[s]?',
            r'authorised\s+dealer',
            r'auth\.\s+dealer',
            r'tractors?',
            r'automobiles?',
            r'agencies',
            r'enterprises?',
            r'farm\s+services?',
            r'motors?'
        ]
        
        candidates = []
        
        # Search for dealer names near keywords
        for result in ocr_results:
            text = result['text']
            
            # Check if text contains dealer keywords
            for keyword in dealer_keywords:
                if re.search(keyword, text, re.IGNORECASE):
                    candidates.append({
                        'text': text,
                        'confidence': result['confidence']
                    })
        
        # Also look for M/s, M/S patterns (common for dealer names)
        ms_pattern = r'M/[Ss]\.?\s*([A-Z][A-Za-z\s&.]+)'
        ms_matches = re.findall(ms_pattern, full_text)
        for match in ms_matches:
            candidates.append({
                'text': match.strip(),
                'confidence': 0.8
            })
        
        # Extract header text (usually contains dealer name)
        header_text = self._extract_header(ocr_results)
        if header_text:
            candidates.append({
                'text': header_text,
                'confidence': 0.7
            })
        
        # Fuzzy match against master
        best_match = self._fuzzy_match_dealer(candidates)
        
        return best_match
    
    def _extract_header(self, ocr_results):
        """Extract header text (top 15% of document)"""
        if not ocr_results:
            return ""
        
        # Find maximum y-coordinate
        max_y = max(
            max(pt[1] for pt in result['bbox'])
            for result in ocr_results if result.get('bbox')
        )
        
        threshold = max_y * 0.15
        
        header_texts = []
        for result in ocr_results:
            bbox = result.get('bbox', [])
            if bbox:
                y_coords = [pt[1] for pt in bbox]
                if min(y_coords) < threshold:
                    header_texts.append(result['text'])
        
        return ' '.join(header_texts)
    
    def _fuzzy_match_dealer(self, candidates):
        """Fuzzy match dealer names against master"""
        if not candidates:
            return {'name': '', 'confidence': 0.0}
        
        if self.dealer_master is None:
            # Return best candidate
            best = max(candidates, key=lambda x: x['confidence'])
            return {'name': best['text'], 'confidence': best['confidence']}
        
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            text = candidate['text']
            
            # Match against master
            for _, row in self.dealer_master.iterrows():
                dealer_name = row['dealer_name']
                variations = row.get('variations', '').split('|')
                
                # Check main name
                score = fuzz.ratio(text.lower(), dealer_name.lower())
                
                # Check variations
                for var in variations:
                    var_score = fuzz.ratio(text.lower(), var.lower())
                    score = max(score, var_score)
                
                if score > best_score:
                    best_score = score
                    best_match = dealer_name
        
        confidence = best_score / 100.0 * candidate['confidence']
        
        return {
            'name': best_match if best_match else candidates[0]['text'],
            'confidence': confidence
        }
    
    def extract_model_name(self, ocr_results, full_text):
        """Extract tractor model name"""
        # Common model patterns
        patterns = [
            r'(?:model|mod|मॉडल)[\s:]*([A-Z]{2,}[\s-]?\d{3,4}[\s]?[A-Z]{0,3})',
            r'\b([A-Z]{2}\s*-?\s*\d{3,4}\s*[A-Z]{0,3})\b',
            r'(?:tractor|ट्रैक्टर)[\s:]*([A-Z]{2,}[\s-]?\d{3,4})',
        ]
        
        candidates = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                model = match.group(1).strip()
                candidates.append(model)
        
        # Exact match against master
        if self.model_master is not None and candidates:
            for candidate in candidates:
                match = self._exact_match_model(candidate)
                if match:
                    return {'name': match, 'confidence': 0.95}
        
        # Return best candidate
        if candidates:
            return {'name': candidates[0], 'confidence': 0.7}
        
        return {'name': '', 'confidence': 0.0}
    
    def _exact_match_model(self, candidate):
        """Exact match model against master"""
        for _, row in self.model_master.iterrows():
            model_name = row['model_name']
            variations = row.get('variations', '').split('|')
            
            # Normalize for comparison
            cand_norm = re.sub(r'\s+', '', candidate.lower())
            model_norm = re.sub(r'\s+', '', model_name.lower())
            
            if cand_norm == model_norm:
                return model_name
            
            for var in variations:
                var_norm = re.sub(r'\s+', '', var.lower())
                if cand_norm == var_norm:
                    return model_name
        
        return None
    
    def extract_horse_power(self, ocr_results, full_text):
        """Extract horse power value"""
        # HP patterns
        patterns = [
            r'(\d+\.?\d*)\s*HP',
            r'HP[\s:]*(\d+\.?\d*)',
            r'horse\s*power[\s:]*(\d+\.?\d*)',
            r'(\d+)\s*एचपी',
        ]
        
        candidates = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                hp = float(match.group(1))
                if 10 <= hp <= 200:  # Reasonable range for tractors
                    candidates.append(hp)
        
        if candidates:
            # Return most common or median value
            return {
                'value': int(np.median(candidates)),
                'confidence': 0.9
            }
        
        return {'value': 0, 'confidence': 0.0}
    
    def extract_asset_cost(self, ocr_results, full_text):
        """Extract asset cost/total amount"""
        # Cost patterns
        patterns = [
            r'total[\s:]*(?:rs\.?|₹)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(?:amount|amt)[\s:]*(?:rs\.?|₹)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(?:cost|price)[\s:]*(?:rs\.?|₹)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(?:rs\.?|₹)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        ]
        
        candidates = []
        
        # Look for amounts in the bottom 30% (where totals usually are)
        bottom_text = self._extract_bottom(ocr_results)
        search_text = bottom_text + " " + full_text
        
        for pattern in patterns:
            matches = re.finditer(pattern, search_text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)
                
                # Reasonable range for tractor cost (₹50k - ₹50L)
                if 50000 <= amount <= 5000000:
                    candidates.append(amount)
        
        if candidates:
            # Return maximum (likely to be total)
            return {
                'value': int(max(candidates)),
                'confidence': 0.85
            }
        
        return {'value': 0, 'confidence': 0.0}
    
    def _extract_bottom(self, ocr_results):
        """Extract text from bottom 30% of document"""
        if not ocr_results:
            return ""
        
        max_y = max(
            max(pt[1] for pt in result['bbox'])
            for result in ocr_results if result.get('bbox')
        )
        
        threshold = max_y * 0.7
        
        bottom_texts = []
        for result in ocr_results:
            bbox = result.get('bbox', [])
            if bbox:
                y_coords = [pt[1] for pt in bbox]
                if min(y_coords) > threshold:
                    bottom_texts.append(result['text'])
        
        return ' '.join(bottom_texts)
    
    def extract_all_fields(self, ocr_results, full_text):
        """Extract all required fields"""
        return {
            'dealer_name': self.extract_dealer_name(ocr_results, full_text),
            'model_name': self.extract_model_name(ocr_results, full_text),
            'horse_power': self.extract_horse_power(ocr_results, full_text),
            'asset_cost': self.extract_asset_cost(ocr_results, full_text)
        }