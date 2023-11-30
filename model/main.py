# Raw copilot output for using resnet50 below

# Step 1: Import necessary libraries
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from load import load_and_preprocess_data
from loss import combined_loss

#Step 1.5: Load data
data = load_and_preprocess_data('Data\images_Y10_test_150.npy')

# Step 2: Load ResNet50 model without top layer
base_model = ResNet50(weights='imagenet', include_top=False)

# Step 3: Add new classification layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Step 5: Compile the model
model.compile(optimizer=Adam(), loss=combined_loss)

# Step 6: Preprocess your training data
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory('data/train', target_size=(224, 224), batch_size=32)

# Step 7: Train the model
model.fit_generator(train_generator, steps_per_epoch=2000, epochs=10)