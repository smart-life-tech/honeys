import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# Constants
SAMPLE_RATE = 16000
DURATION = 10  # Duration of each audio clip in seconds
NUM_CLASSES = 3  # Number of hive states (e.g., healthy, queenless, sick)
LOG_DIR = "logs/"

# Function to preprocess audio data
def preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    # Extract Mel-frequency cepstral coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
    return mfccs

# Load data and labels
def load_data(data_dir):
    data = []
    labels = []
    classes = os.listdir(data_dir)
    for i, cls in enumerate(classes):
        class_dir = os.path.join(data_dir, cls)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            mfccs = preprocess_audio(file_path)
            data.append(mfccs)
            labels.append(i)
    return np.array(data), np.array(labels)

# Split data into training and validation sets
def split_data(data, labels):
    return train_test_split(data, labels, test_size=0.2, random_state=42)

# Build deep learning model
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main function for training the model
def train_model(data_dir):
    # Load and preprocess data
    data, labels = load_data(data_dir)
    input_shape = data[0].shape
    data_train, data_val, labels_train, labels_val = split_data(data, labels)

    # Convert labels to one-hot encoding
    labels_train = to_categorical(labels_train, num_classes=NUM_CLASSES)
    labels_val = to_categorical(labels_val, num_classes=NUM_CLASSES)

    # Build and train the model
    model = build_model(input_shape)
    callbacks = [
        ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        TensorBoard(log_dir=LOG_DIR)
    ]
    history = model.fit(data_train, labels_train, epochs=20, batch_size=32, validation_data=(data_val, labels_val), callbacks=callbacks)

    # Evaluate the model
    loss, accuracy = model.evaluate(data_val, labels_val)
    print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')

    # Save the trained model
    model.save('hive_state_detection_model.h5')
    return history

# Function for making predictions
def predict_hive_state(audio_file):
    # Load the trained model
    model = load_model('hive_state_detection_model.h5')

    # Preprocess the audio file
    mfccs = preprocess_audio(audio_file)

    # Perform inference
    prediction = model.predict(np.expand_dims(mfccs, axis=0))
    predicted_class = np.argmax(prediction)
    return predicted_class

# Example usage
if __name__ == '__main__':
    data_dir = 'hive_audio_data'
    history = train_model(data_dir)
    predicted_class = predict_hive_state('test_audio.wav')
    print(f'Predicted hive state: {predicted_class}')
