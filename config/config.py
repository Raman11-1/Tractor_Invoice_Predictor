"""
config/config.py - Configuration Parameters
"""

import os
from pathlib import Path

class Config:
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    # Master files
    DEALER_MASTER = DATA_DIR / "master_files" / "dealer_master.csv"
    MODEL_MASTER = DATA_DIR / "master_files" / "model_master.csv"
    
    # OCR Configuration
    OCR_ENGINES = ['paddleocr', 'easyocr', 'tesseract']
    OCR_LANGUAGES = ['en', 'hi']  # English, Hindi
    TESSERACT_CONFIG = '--oem 3 --psm 6'
    
    # Tesseract path for Windows (update if your installation path is different)
    TESSERACT_CMD = r"D:\Visual_Computing\Tesseract-ocr\tesseract.exe"
    
    # Poppler path for Windows pdf2image (update if your installation path is different)
    POPPLER_PATH = r'C:\Program Files\poppler\Library\bin'
    
    # Vision Model Configuration
    USE_VLM = False  # Set to False to avoid downloading large models
    VLM_MODEL = "Qwen/Qwen2-VL-2B-Instruct"  # Lightweight VLM
    
    # Signature/Stamp Detection
    SIGNATURE_MODEL = "yolov8n"  # Lightweight YOLO
    STAMP_CONFIDENCE_THRESHOLD = 0.5
    SIGNATURE_CONFIDENCE_THRESHOLD = 0.5
    USE_YOLO = False  # Set to False to use CV-based detection (no model download needed)
    
    # Field Extraction
    FUZZY_MATCH_THRESHOLD = 90  # For dealer name
    NUMERIC_TOLERANCE = 0.05  # 5% tolerance for HP and Cost
    
    # Performance Targets
    TARGET_DLA = 0.95  # 95% Document-Level Accuracy
    MAX_LATENCY_SEC = 30
    MAX_COST_USD = 0.01
    
    # Processing
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    DEVICE = 'cpu'  # Use CPU for compatibility
    
    # Image Preprocessing
    IMAGE_SIZE = (1024, 1024)
    DPI = 300
    
    # Output Format
    OUTPUT_FIELDS = [
        'dealer_name',
        'model_name', 
        'horse_power',
        'asset_cost',
        'signature',
        'stamp'
    ]
    
    # Confidence Scoring
    MIN_CONFIDENCE = 0.7
    
    # Cost Estimation (per 1000 tokens/images)
    COST_PADDLEOCR = 0.0  # Free/Open-source
    COST_EASYOCR = 0.0
    COST_VLM = 0.0001  # Local inference
    COST_YOLO = 0.0
    
    @classmethod
    def get_cost_estimate(cls, doc_length=1):
        """Estimate processing cost per document"""
        base_cost = cls.COST_PADDLEOCR + cls.COST_EASYOCR
        if cls.USE_VLM:
            base_cost += cls.COST_VLM
        if cls.USE_YOLO:
            base_cost += cls.COST_YOLO
        return base_cost * doc_length
    
    @classmethod
    def setup_tesseract(cls):
        """Setup Tesseract path for pytesseract"""
        try:
            import pytesseract
            if os.path.exists(cls.TESSERACT_CMD):
                pytesseract.pytesseract.tesseract_cmd = cls.TESSERACT_CMD
                return True
            else:
                print(f"Warning: Tesseract not found at {cls.TESSERACT_CMD}")
                print("Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
                return False
        except ImportError:
            print("Warning: pytesseract not installed")
            return False
    
    @classmethod
    def verify_setup(cls):
        """Verify all paths and dependencies"""
        issues = []
        
        # Check directories
        if not cls.DATA_DIR.exists():
            issues.append(f"Data directory not found: {cls.DATA_DIR}")
            cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Created: {cls.DATA_DIR}")
        
        if not cls.OUTPUT_DIR.exists():
            issues.append(f"Output directory not found: {cls.OUTPUT_DIR}")
            cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Created: {cls.OUTPUT_DIR}")
        
        # Check master files
        if not cls.DEALER_MASTER.exists():
            issues.append(f"Dealer master file not found: {cls.DEALER_MASTER}")
        
        if not cls.MODEL_MASTER.exists():
            issues.append(f"Model master file not found: {cls.MODEL_MASTER}")
        
        # Check Tesseract
        if not os.path.exists(cls.TESSERACT_CMD):
            issues.append(f"Tesseract not found at: {cls.TESSERACT_CMD}")
        
        # Check Poppler
        if not os.path.exists(cls.POPPLER_PATH):
            issues.append(f"Poppler not found at: {cls.POPPLER_PATH}")
        
        if issues:
            print("\n⚠️  Configuration Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
            print("\nPlease resolve these issues before running the system.\n")
            return False
        else:
            print("✅ Configuration verified successfully!")
            return True