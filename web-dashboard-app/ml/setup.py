#!/usr/bin/env python3
"""
ML Environment Setup Script
This script sets up the Python environment for machine learning analysis.
It installs required packages and configures the environment.
"""

import subprocess
import sys
import os
import platform
import json
from pathlib import Path
import shutil

# Define dependencies
BASIC_DEPS = [
    "pandas",
    "numpy",
    "scikit-learn",
    "nltk",
    "spacy",
    "requests",
    "python-dotenv",
    "matplotlib"
]

ADVANCED_DEPS = [
    "plotly",
    "wordcloud",
]

# Configuration
ML_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = ML_DIR / "output"
ENV_FILE = ML_DIR / ".env"
PLOTS_DIR = OUTPUT_DIR / "plots"

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def run_command(cmd, description):
    """Run a shell command with proper error handling"""
    print(f"{description}...")
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(f"✓ Success: {description}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description} failed")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible"""
    print_section("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Warning: Python 3.8+ is recommended for this application")
        if input("Continue anyway? (y/N): ").lower() != 'y':
            sys.exit(1)
    else:
        print("✓ Python version is compatible")

def setup_directories():
    """Set up necessary directories"""
    print_section("Setting Up Directories")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"✓ Created output directory: {OUTPUT_DIR}")
    
    # Create plots directory
    PLOTS_DIR.mkdir(exist_ok=True)
    print(f"✓ Created plots directory: {PLOTS_DIR}")

def install_dependencies():
    """Install required Python packages"""
    print_section("Installing Dependencies")
    
    # Install basic dependencies
    print("Installing basic machine learning packages...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade"] + BASIC_DEPS, 
                "Installing basic dependencies")
    
    # Install advanced dependencies
    print("\nInstalling advanced visualization packages...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade"] + ADVANCED_DEPS, 
                "Installing advanced dependencies")
    
    # Install spaCy Spanish model
    print("\nInstalling Spanish language model for spaCy...")
    run_command([sys.executable, "-m", "spacy", "download", "es_core_news_sm"], 
                "Installing Spanish language model")
    
    # Download NLTK data
    print("\nDownloading NLTK data...")
    run_command([sys.executable, "-c", 
                "import nltk; nltk.download('punkt'); nltk.download('stopwords')"], 
                "Downloading NLTK data")

def check_env_file():
    """Check if .env file exists and has required variables"""
    print_section("Checking Environment Configuration")
    
    if not ENV_FILE.exists():
        print("❌ .env file not found")
        create_env = input("Create a sample .env file? (Y/n): ")
        if create_env.lower() != 'n':
            with open(ENV_FILE, 'w') as f:
                f.write("SUPABASE_URL=https://your-project-id.supabase.co\n")
                f.write("SUPABASE_KEY=your-supabase-service-key\n")
            print(f"✓ Created sample .env file at {ENV_FILE}")
            print("⚠️ Please update with your actual Supabase credentials")
        return
    
    # Check if .env file has required variables
    with open(ENV_FILE, 'r') as f:
        env_content = f.read()
    
    if "SUPABASE_URL" not in env_content or "SUPABASE_KEY" not in env_content:
        print("❌ .env file is missing required variables")
        print("⚠️ Please ensure SUPABASE_URL and SUPABASE_KEY are defined")
    else:
        print("✓ .env file exists with required variables")

def test_ml_scripts():
    """Test that ML scripts can be imported without errors"""
    print_section("Testing ML Scripts")
    
    test_script = """
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try importing from sentiment_analysis
    from ml.sentiment_analysis import preprocess_text
    print("Successfully imported from sentiment_analysis")
    
    # Try importing from topic_analysis
    from ml.topic_analysis import preprocess_text
    print("Successfully imported from topic_analysis")
    
    print("All imports successful!")
except Exception as e:
    print(f"Error importing ML scripts: {str(e)}")
    sys.exit(1)
"""
    
    test_file = ML_DIR / "test_imports.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    result = run_command([sys.executable, str(test_file)], "Testing ML script imports")
    test_file.unlink()  # Remove test file
    
    if result and "All imports successful" in result:
        print("✓ ML scripts can be imported successfully")
    else:
        print("❌ Error importing ML scripts")

def main():
    """Main setup function"""
    print_section("ML Environment Setup")
    print("This script will set up the Python environment for ML analysis")
    
    # Check Python version
    check_python_version()
    
    # Setup directories
    setup_directories()
    
    # Install dependencies
    install_dependencies()
    
    # Check .env file
    check_env_file()
    
    # Test ML scripts
    test_ml_scripts()
    
    print_section("Setup Complete")
    print("The ML environment has been set up successfully!")
    print("\nTo run sentiment analysis:")
    print(f"  python {ML_DIR}/sentiment_analysis.py")
    print("\nTo run topic analysis:")
    print(f"  python {ML_DIR}/topic_analysis.py")
    print("\nResults will be saved to:")
    print(f"  {OUTPUT_DIR}")

if __name__ == "__main__":
    main()