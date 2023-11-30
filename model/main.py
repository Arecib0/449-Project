# Raw copilot output for using resnet50 below

# Step 1: Import necessary libraries
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# Step 2: Load ResNet50 model without top layer
base_model = ResNet50(weights='imagenet', include_top=False)

# Step 3: Add new classification layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(number_of_classes, activation='softmax')(x)  # replace number_of_classes with your number of classes

model = Model(inputs=base_model.input, outputs=predictions)

# Step 4: Freeze the layers of ResNet50 (optional)
for layer in base_model.layers:
    layer.trainable = False

# Step 5: Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy')

# Step 6: Preprocess your training data
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory('data/train', target_size=(224, 224), batch_size=32)

# Step 7: Train the model
model.fit_generator(train_generator, steps_per_epoch=2000, epochs=10)