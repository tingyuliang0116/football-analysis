import pickle

def save_detections(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

# Function to load detections
def load_detections(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)