# train_scratch.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_loader import train_generator, val_generator

# Add test generator (since data_loader.py doesn't have it)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'data/Testing',
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Build the scratch model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('models/scratch_model.keras', monitor='val_accuracy', save_best_only=True)

# Train (all arguments after first must be keyword arguments)
history = model.fit(
    train_generator, 
    epochs=30,                     
    callbacks=[early_stop, checkpoint],  
    verbose=1                      
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"Scratch CNN - Test accuracy: {test_acc:.4f}")