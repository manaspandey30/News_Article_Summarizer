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

def check_transformers():
    """Check if transformers and torch are available"""
    try:
        import transformers
        import torch
        print("Transformers and PyTorch are available.")
        return True
    except ImportError:
        print("Transformers or PyTorch not available.")
        return False

def check_basic_deps():
    """Check if basic dependencies are available"""
    try:
        import flask
        import requests
        import nltk
        import bs4
        return True
    except ImportError as e:
        print(f"Missing basic dependency: {e}")
        return False

def start_app(use_fallback=False):
    """Start the appropriate version of the application"""
    script = "fallback.py" if use_fallback else "app.py"
    
    # Start the Flask app in a separate thread
    print(f"Starting backend server using {script}...")
    flask_thread = threading.Thread(target=lambda: os.system(f'python {script}'))
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
    
    # Check if basic dependencies are available
    if not check_basic_deps():
        print("Basic dependencies are missing.")
        install = input("Would you like to install basic dependencies? (y/n): ")
        if install.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install", "flask==2.0.1", "requests==2.26.0", "beautifulsoup4==4.10.0", "nltk==3.6.5", "flask-cors==3.0.10"])
        else:
            print("Exiting: basic dependencies are required to run the application.")
            sys.exit(1)
    
    # Check for transformers
    has_transformers = check_transformers()
    
    if not has_transformers:
        print("\n" + "="*80)
        print("Advanced summarization (transformers) is not available.")
        print("You can run with basic summarization or install advanced dependencies.")
        print("="*80 + "\n")
        
        choice = input("How would you like to proceed?\n1. Run with basic summarization\n2. Install advanced dependencies and run full version\n3. Exit\nChoice (1/2/3): ")
        
        if choice == "1":
            start_app(use_fallback=True)
        elif choice == "2":
            print("Installing transformers and PyTorch...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.12.5", "torch"])
                print("Installation successful, starting full version...")
                start_app(use_fallback=False)
            except Exception as e:
                print(f"Installation failed: {str(e)}")
                print("Starting with basic summarization instead...")
                start_app(use_fallback=True)
        else:
            print("Exiting.")
            sys.exit(0)
    else:
        # Has transformers, run full version
        start_app(use_fallback=False) 