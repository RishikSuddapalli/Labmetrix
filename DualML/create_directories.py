# create_directories.py
import os

def create_project_directories():
    directories = [
        'data',
        'models/classification',
        'models/regression',
        'mlruns',
        'logs',
        'tmp'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    create_project_directories()