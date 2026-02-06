import os
import uvicorn
import subprocess
import sys

def main():
    print("Starting Image Super-Resolution App...")
    print("Backend and Frontend will be available at http://localhost:8000")
    
    # Ensure we are running from the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Path to main.py
    backend_dir = os.path.join(script_dir, "webapp", "backend")
    
    if not os.path.exists(backend_dir):
        print(f"Error: Could not find backend directory at {backend_dir}")
        return

    # Add backend to python path so it can import model
    sys.path.append(backend_dir)
    
    # Change into backend dir to make relative paths in main.py work easily
    # (Although main.py uses os.path.dirname, uvicorn might care about CWD for reloading)
    os.chdir(backend_dir)

    # Run Uvicorn
    # We run it as a module to allow reloading if needed, but here we just call run directly
    try:
        # Import main here to start the app
        # or use uvicorn.run command line style
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\nStopping app...")

if __name__ == "__main__":
    main()
