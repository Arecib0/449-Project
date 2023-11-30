import numpy as np

def load_and_preprocess_data(filepath):
    # Load the .npy file
    data = np.load(filepath)

    # Assuming your data is in the format (color, height, width)
    # Reshape it to (height, width, color)
    data = np.transpose(data, (1, 2, 0))

    # Normalize the data
    data = data / 255.0

    return data