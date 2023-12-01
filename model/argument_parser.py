import argparse

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Train a ResNet50 model.')
    parser.add_argument('--data_path', type=str, default='Data\images_Y10_test_150.npy', help='Path to the training data.')
    parser.add_argument('--labels_path', type=str, default='Data\labels_train_150.npy', help='Path to the training labels.')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    return parser