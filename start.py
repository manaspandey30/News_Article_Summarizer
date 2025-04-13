import os
import webbrowser
import threading
import time
import subprocess
import sys
import platform

def open_browser():
    """Wait for Flask to start and then open the browser"""
    time.sleep(2)  # Give Flask some time to start up
    print("Opening application in browser...")
    # Get the absolute path to index.html
    abs_path = os.path.abspath("index.html")
    webbrowser.open('file://' + abs_path)

def install_dependencies(use_fallback=False):
    """Install dependencies with appropriate options"""
    try:
        print("Installing dependencies...")
        
        if use_fallback:
            # Try installing without torch as a fallback
            print("Using fallback installation without specific torch version...")
            with open("requirements_simple.txt", "w") as f:
                f.write("flask==2.0.1\n")
                f.write("flask-cors==3.0.10\n")
                f.write("requests==2.26.0\n")
                f.write("beautifulsoup4==4.10.0\n")
                f.write("nltk==3.6.5\n")
                f.write("transformers==4.12.5\n")
                f.write("numpy==1.21.4\n")
                f.write("scikit-learn==1.0.1\n")
            
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_simple.txt"])
            
            # Try to install torch separately
            try:
                print("Attempting to install PyTorch separately...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
            except:
                print("Could not install PyTorch. Some functionality may be limited.")
        else:
            # Normal installation
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        return True
    except Exception as e:
        print(f"Installation error: {str(e)}")
        if not use_fallback:
            print("Trying fallback installation method...")
            return install_dependencies(use_fallback=True)
        return False

def start_app():
    """Start the Flask backend and open the frontend"""
    # Check if the required packages are installed
    try:
        print("Checking dependencies...")
        # Try importing key packages
        import flask
        try:
            import transformers
            import torch
            print("All required packages are installed (including transformers and torch).")
        except ImportError as e:
            print(f"Warning: {e}")
            print("Basic functionality will work, but summarization may be limited.")
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        install = input("Would you like to install missing dependencies? (y/n): ")
        if install.lower() == 'y':
            if not install_dependencies():
                print("\nWARNING: Some dependencies could not be installed.")
                print("The application will start but may have limited functionality.")
                print("You can continue with basic features or exit and try again later.")
                
                proceed = input("Would you like to proceed anyway? (y/n): ")
                if proceed.lower() != 'y':
                    print("Exiting application.")
                    return
        else:
            print("Exiting: dependencies are required to run the application.")
            return

    # Start the Flask app in a separate thread
    print("Starting backend server...")
    flask_thread = threading.Thread(target=lambda: os.system('python app.py'))
    flask_thread.daemon = True
    flask_thread.start()

    # Open the browser in the main thread after a delay
    open_browser()
    
    # Keep the script running
    try:
        while True:
            # Check if the Flask thread is still running
            if not flask_thread.is_alive():
                print("Backend server stopped. Exiting...")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down the application...")
        # On Windows, we need to forcefully terminate the Flask process
        if platform.system() == 'Windows':
            os.system('taskkill /f /im python.exe')
        # On Unix-like systems, the KeyboardInterrupt should propagate to the Flask process

if __name__ == '__main__':
    print("Starting Smart Article Summarizer...")
    start_app() 