import argparse

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Train a ResNet50 model.')
    parser.add_argument('--data_path', type=str, default='Data\images_Y10_test_150.npy', help='Path to the training data.')
    parser.add_argument('--labels_path', type=str, default='Data\labels_train_150.npy', help='Path to the training labels.')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--rho', type=float, default=0.5, help='Target entropy.')
    parser.add_argument('--m', type=float, default=0.1, help='Entropy threshold.')
    parser.add_argument('--loss_weight', type=float, default=0.005, help='Weight for clustering and entropy separation losses.')
    return parser