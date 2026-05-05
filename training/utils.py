# utils.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_dir='data/Training', 
                      test_dir='data/Testing',
                      target_size=(128,128), 
                      batch_size=32,
                      validation_split=0.2,
                      augment=False):
    """
    Creates train, validation, and test generators.
    
    Args:
        train_dir: path to training folder (with class subfolders)
        test_dir: path to testing folder
        target_size: tuple (height, width)
        batch_size: int
        validation_split: fraction of training data to use for validation
        augment: if True, applies augmentation to training data
    
    Returns:
        train_generator, val_generator, test_generator
    """
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            validation_split=validation_split
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
    
    # Validation and test generators never get augmentation
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen