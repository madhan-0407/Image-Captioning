import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# --- Constants and Configurations ---
tokenizer_path = '/home/vglug/Documents/Image-Captioning/tokenizer.pkl'
sequences_path = '/home/vglug/Documents/Image-Captioning/sequences.pkl'
image_feature_path = "/home/vglug/Documents/Image-Captioning/image_features.pkl"
max_caption_length = 36
embedding_dim = 256
lstm_units = 256
dropout_rate = 0.5
epochs = 20
batch_size = 64

def create_model(vocab_size):
  # Image feature encoder
    inputs1 = Input(shape=(2048,))
    fe1 = Dense(embedding_dim, activation='relu')(inputs1)
    fe2 = Dropout(dropout_rate)(fe1)
    fe3 = Dense(lstm_units, activation = 'relu')(fe2)

    # Text sequence encoder
    inputs2 = Input(shape=(max_caption_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(dropout_rate)(se1)
    se3 = LSTM(lstm_units)(se2)

    # Decoder
    decoder1 = add([fe3, se3])
    decoder2 = Dense(lstm_units, activation='relu')(decoder1)
    decoder3 = Dropout(dropout_rate)(decoder2)
    outputs = Dense(vocab_size, activation='softmax')(decoder3)

    # Tie it together
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    return model

def data_preparation(sequences, tokenizer, image_features, max_caption_length = 36):
    X2_list = []
    y_list = []
    image_input_array_list = []

    vocab_size = len(tokenizer.word_index) + 1
    for image_id, seq_list in sequences.items():
            if image_id not in image_features:
                  continue
            img_features = image_features[image_id]
            for seq in seq_list:
              for i in range(1, len(seq)):
                image_input_array_list.append(img_features)
                X2_list.append(seq[:i])
                y_list.append(seq[i])
    
    X2 = pad_sequences(X2_list, maxlen=max_caption_length, padding='post')
    y = to_categorical(y_list, num_classes=vocab_size)
    image_input_array = np.array(image_input_array_list)

    return X2, y, image_input_array

# --- Main Execution ---
print("Loading data...")
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
with open(sequences_path, 'rb') as f:
    sequences = pickle.load(f)
with open(image_feature_path, 'rb') as f:
    image_features = pickle.load(f)
print("Data loaded.\n")

X2, y, image_input_array = data_preparation(sequences, tokenizer, image_features)
print("Data preparation for training done.\n")

# Split the data for training and testing
X_train, X_test, y_train, y_test, image_input_array_train, image_input_array_test = train_test_split(X2, y, image_input_array, test_size=0.2, random_state=42)

vocab_size = len(tokenizer.word_index) + 1
model = create_model(vocab_size)
print("Model created.\n")
checkpoint = ModelCheckpoint(
    "/home/vglug/Documents/Image-Captioning/image_captioning_best.h5",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    save_weights_only=False
)

print("Starting model training...")
model.fit([image_input_array_train, X_train], y_train, epochs = epochs, validation_data = ([image_input_array_test, X_test], y_test), callbacks = [checkpoint], batch_size=batch_size)
print("Model training completed.\n")