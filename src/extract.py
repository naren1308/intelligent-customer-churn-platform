import os
import sys

try:
    import py7zr
except ImportError:
    print("Please install py7zr first by running: pip install py7zr")
    sys.exit(1)

def extract_7z_files(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.7z')]
    
    if not files:
        print("No .7z files found in the data directory.")
        return

    for file in files:
        file_path = os.path.join(data_dir, file)
        print(f"Extracting {file}...")
        try:
            with py7zr.SevenZipFile(file_path, mode='r') as z:
                z.extractall(path=data_dir)
            print(f"Successfully extracted {file}")
        except Exception as e:
            print(f"Error extracting {file}: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.abspath(os.path.join(current_dir, '../data'))
    extract_7z_files(DATA_DIR)
