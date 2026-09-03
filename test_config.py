from config.config import Config

# Test configuration
print("Testing configuration...")
print(f"Base Directory: {Config.BASE_DIR}")
print(f"Data Directory: {Config.DATA_DIR}")
print(f"Output Directory: {Config.OUTPUT_DIR}")

# Setup Tesseract
Config.setup_tesseract()

# Verify setup
Config.verify_setup()