# %%
!pip install tensorflow

# %%
import random
random.seed(0)

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.random.set_seed(0)


# %%
import os
import json
from zipfile import ZipFile
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization

# %%
print(os.listdir("../archive"))

print(len(os.listdir("../archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train")))
print(os.listdir("../archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train")[:5])

# %%
print(len(os.listdir("../archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___Apple_scab")))
print(os.listdir("../archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___Apple_scab")[:5])

# %%
base_dir = "../archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"

# %%
image_path="../archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417_90deg.JPG"

img = mpimg.imread(image_path)

print(img.shape)

plt.imshow(img)
plt.axis('off')
plt.show()

# %%
img_size =224
batch_size = 32

# %%
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# %%
train_generator =data_gen.flow_from_directory(
    base_dir,
    target_size = (img_size,img_size),
    batch_size = batch_size,
    subset = "training",
    class_mode = "categorical"
)

# %%
# Check data balance
import matplotlib.pyplot as plt

class_counts = {}
for class_name in train_generator.class_indices.keys():
    class_dir = os.path.join(base_dir, class_name)
    count = len(os.listdir(class_dir))
    class_counts[class_name] = count

# Plot distribution
plt.figure(figsize=(15, 8))
plt.bar(range(len(class_counts)), list(class_counts.values()))
plt.xticks(range(len(class_counts)), list(class_counts.keys()), rotation=45, ha='right')
plt.ylabel('Number of Images')
plt.title('Training Data Distribution by Class')
plt.tight_layout()
plt.show()

print("Class Distribution:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images")
print(f"\nTotal images: {sum(class_counts.values())}")
print(f"Min: {min(class_counts.values())}, Max: {max(class_counts.values())}")

# %%
valid_dir = "../archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    valid_dir,
    target_size = (img_size,img_size),
    batch_size = batch_size,
    class_mode = "categorical"
)

# %%
# Load MobileNetV2 as base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze the base model initially

# Build the model
model = models.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    layers.Dense(256, activation='relu'),
    Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# %%
model.summary()

# %%
model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

# %%
history = model.fit(
    train_generator,
    steps_per_epoch  = train_generator.samples//batch_size,
    epochs = 20,
    validation_data = validation_generator ,
    validation_steps=  validation_generator.samples//batch_size
)

# %%
print("evaluating model ..")
val_loss,val_accuracy = model.evaluate(validation_generator,steps = validation_generator.samples//batch_size)
print(f"validation accuracy : {val_accuracy*100:.2f}%")
print(f"validation loss : {val_loss:.4f}")

# Get final training accuracy
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"\nFinal Training Accuracy: {final_train_acc*100:.2f}%")
print(f"Final Validation Accuracy: {final_val_acc*100:.2f}%")
print(f"Overfitting indicator: {(final_train_acc - final_val_acc)*100:.2f}%")

# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model_accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model_loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc = 'upper left')
plt.show()

# %%
print("Training History:")
for key, values in history.history.items():
    print(f"{key}: {values}")

# %%
def load_and_preprocess_image(image_path,target_size = (224,224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array,axis = 0)
    img_array = img_array.astype('float32')/255
    return img_array
def predict_image_class(model,image_path,class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions,axis = 1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class_name, confidence

# %%
class_indices = {v :k for k,v in train_generator.class_indices.items()}

# %%
class_indices


