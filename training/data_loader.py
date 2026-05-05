from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create data preprocessor: normalizes pixel values (0-255 → 0-1) and reserves 20% for validation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Generator for training data (80% of images from 'data/Training')
train_generator = datagen.flow_from_directory(
    'data/Training',        # folder with class subfolders (glioma, meningioma, etc.)
    target_size=(128,128),  # resize all images to 128x128 pixels
    batch_size=32,          # yield 32 images per batch
    class_mode='categorical',  # one-hot labels for 4 classes
    subset='training'       # use the 80% training portion
)

# Generator for validation data (remaining 20% of images)
val_generator = datagen.flow_from_directory(
    'data/Training',
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'     # use the 20% validation portion
)