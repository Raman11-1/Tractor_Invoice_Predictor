"""
executable.py - Main execution script for IDFC Hackathon
"""

import argparse
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from config.config import Config
from utils.preprocessing import DocumentPreprocessor
from utils.ocr_engine import MultiOCREngine
from utils.field_extraction import FieldExtractor
from utils.signature_detection import SignatureStampDetector

class DocumentAISystem:
    """Complete Document AI System"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        
        # Initialize components
        print("Initializing Document AI System...")
        
        self.preprocessor = DocumentPreprocessor(
            dpi=self.config.DPI,
            target_size=self.config.IMAGE_SIZE
        )
        
        self.ocr_engine = MultiOCREngine(
            languages=self.config.OCR_LANGUAGES,
            use_gpu=(self.config.DEVICE == 'cuda')
        )
        
        self.field_extractor = FieldExtractor(
            dealer_master_path=self.config.DEALER_MASTER,
            model_master_path=self.config.MODEL_MASTER
        )
        
        self.signature_detector = SignatureStampDetector(
            use_yolo=False  # Use CV-based detection (no training required)
        )
        
        print("System initialized successfully!")
    
    def process_document(self, file_path):
        """Process a single document"""
        start_time = time.time()
        
        try:
            # 1. Preprocess
            print(f"Processing: {file_path}")
            processed_images = self.preprocessor.preprocess(file_path)
            
            # Use first page (assuming single-page quotations)
            image_data = processed_images[0]
            original_image = image_data['original']
            processed_image = image_data['processed']
            
            # 2. OCR Extraction
            print("  - Running OCR...")
            ocr_data = self.ocr_engine.extract_structured_data(processed_image)
            ocr_results = ocr_data['ocr_results']
            full_text = ocr_data['full_text']
            
            # 3. Field Extraction
            print("  - Extracting fields...")
            fields = self.field_extractor.extract_all_fields(ocr_results, full_text)
            
            # 4. Signature & Stamp Detection
            print("  - Detecting signatures/stamps...")
            detections = self.signature_detector.detect(original_image)
            signature, stamp = self.signature_detector.format_detection(detections)
            
            # 5. Calculate confidence
            confidences = [
                fields['dealer_name']['confidence'],
                fields['model_name']['confidence'],
                fields['horse_power']['confidence'],
                fields['asset_cost']['confidence'],
                signature['confidence'],
                stamp['confidence']
            ]
            overall_confidence = sum(confidences) / len(confidences)
            
            # 6. Processing time
            processing_time = time.time() - start_time
            
            # 7. Cost estimate
            cost_estimate = self.config.get_cost_estimate()
            
            # 8. Format output
            result = {
                'doc_id': Path(file_path).stem,
                'fields': {
                    'dealer_name': fields['dealer_name'].get('name', ''),
                    'model_name': fields['model_name'].get('name', ''),
                    'horse_power': fields['horse_power'].get('value', 0),
                    'asset_cost': fields['asset_cost'].get('value', 0),
                    'signature': {
                        'present': signature['present'],
                        'bbox': signature['bbox']
                    },
                    'stamp': {
                        'present': stamp['present'],
                        'bbox': stamp['bbox']
                    }
                },
                'confidence': round(overall_confidence, 2),
                'processing_time_sec': round(processing_time, 2),
                'cost_estimate_usd': cost_estimate
            }
            
            print(f"  ✓ Completed in {processing_time:.2f}s (confidence: {overall_confidence:.2%})")
            
            return result
            
        except Exception as e:
            print(f"  ✗ Error processing {file_path}: {e}")
            return {
                'doc_id': Path(file_path).stem,
                'error': str(e),
                'fields': {
                    'dealer_name': '',
                    'model_name': '',
                    'horse_power': 0,
                    'asset_cost': 0,
                    'signature': {'present': False, 'bbox': [0, 0, 0, 0]},
                    'stamp': {'present': False, 'bbox': [0, 0, 0, 0]}
                },
                'confidence': 0.0,
                'processing_time_sec': 0.0,
                'cost_estimate_usd': 0.0
            }
    
    def process_batch(self, input_dir, output_dir):
        """Process multiple documents"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF and image files
        file_patterns = ['*.pdf', '*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        files = []
        for pattern in file_patterns:
            files.extend(input_path.glob(pattern))
        
        print(f"\nFound {len(files)} documents to process")
        print("=" * 60)
        
        results = []
        total_time = 0
        total_cost = 0
        
        for file_path in files:
            result = self.process_document(file_path)
            results.append(result)
            
            total_time += result['processing_time_sec']
            total_cost += result['cost_estimate_usd']
            
            # Save individual result
            output_file = output_path / f"{result['doc_id']}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        # Save combined results
        combined_output = output_path / 'all_results.json'
        with open(combined_output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total documents: {len(results)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time: {total_time/len(results):.2f}s per document")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Average cost: ${total_cost/len(results):.6f} per document")
        
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"Average confidence: {avg_confidence:.2%}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='IDFC Document AI System')
    parser.add_argument('--input_dir', type=str, default='data/sample_pdfs',
                       help='Input directory containing PDFs/images')
    parser.add_argument('--output_dir', type=str, default='outputs/results',
                       help='Output directory for results')
    parser.add_argument('--single_file', type=str, default=None,
                       help='Process a single file')
    
    args = parser.parse_args()
    
    # Initialize system
    system = DocumentAISystem()
    
    if args.single_file:
        # Process single file
        result = system.process_document(args.single_file)
        print("\nResult:")
        print(json.dumps(result, indent=2))
    else:
        # Process batch
        system.process_batch(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()