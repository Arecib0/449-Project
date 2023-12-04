# PHYS449
Danny Meng
Ethan Lousenberg 20891205
Eric McFarlane

## Dependencies
- PyTorch
- numpy
- PIL
- Torchvision
- memory
- yaml
- matplotlib

## Description of Task
In this task we use a Semi Supervised Convolutional Neural Network using ResNet50 and SGD with Domain Adaptation to classify galaxy images into 1 of 3 classes. We than output a plot of the 3 types of losses used; Cross Entropy, Adaptive Clustering and Entropy Seperation vs Epochs and to better demonstrate the usefulness of Domain Adaptation in a CNN for Galaxy classification our target dataset has labels that are not used during training but are used to calculate the accuracy of our model vs epoch as it trains on the target set, which is outputted as a plot.

## Use of AI 
Git CoPilot was used heavily to make suggestions and implement code. ChatGPT was used to check theory. 

## Running 
To run this program, please edit 'Config.yaml' contained in the Data Directory with the file path of the data you wish to use. Hyperparameters can also be edited from this file at user discretion.

To run `main.py`, use

```sh
python model\main.py -
```
in terminal.
