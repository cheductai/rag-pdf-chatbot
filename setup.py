"""Setup script for RAG PDF Chatbot."""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def create_virtual_environment():
    """Create virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        return False

def get_activation_command():
    """Get the appropriate activation command for the platform."""
    if os.name == 'nt':  # Windows
        return "venv\\Scripts\\activate"
    else:  # Unix-like (Linux, macOS)
        return "source venv/bin/activate"

def install_dependencies():
    """Install required dependencies."""
    try:
        print("📦 Installing dependencies...")
        
        # Determine pip executable path
        if os.name == 'nt':  # Windows
            pip_executable = "venv\\Scripts\\pip"
        else:  # Unix-like
            pip_executable = "venv/bin/pip"
        
        # Install requirements
        subprocess.run([pip_executable, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("💡 Try installing manually with: pip install -r requirements.txt")
        return False

def setup_environment_file():
    """Set up .env file from example."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not env_example.exists():
        print("❌ .env.example file not found")
        return False
    
    try:
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from example")
        print("⚠️  Please edit .env file and add your OpenAI API key")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/uploads",
        "data/processed", 
        "data/faiss_index"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Created necessary directories")
    return True

def main():
    """Main setup function."""
    print("🚀 RAG PDF Chatbot Setup")
    print("=" * 30)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Create virtual environment
    if success and not create_virtual_environment():
        success = False
    
    # Install dependencies
    if success and not install_dependencies():
        success = False
    
    # Setup environment file
    if success and not setup_environment_file():
        success = False
    
    # Create directories
    if success and not create_directories():
        success = False
    
    print("\n" + "=" * 30)
    
    if success:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Activate virtual environment:")
        print(f"   {get_activation_command()}")
        print("2. Edit .env file and add your OpenAI API key")
        print("3. Run the application:")
        print("   python main.py")
        print("\n💡 Make sure you have an OpenAI API key!")
    else:
        print("❌ Setup failed. Please check the errors above.")
        print("💡 You may need to install dependencies manually.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
