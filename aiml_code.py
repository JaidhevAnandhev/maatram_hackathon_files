import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import sys
import logging


# Define the model
def create_model():
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_length))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def getModelReadyForPrediction():
# Set up logging
    logging.basicConfig(filename='anomaly_detection.log', level=logging.INFO)

# Load the dataset
data = pd.read_csv('text_data.csv')  # Ensure this path is correct

# Encode labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# Convert texts to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform input size
max_length = 100  # You can adjust this based on your data
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

# Check if model already exists
model_file = 'model.h5'
if os.path.exists(model_file):
    model = load_model(model_file)  # Load the entire model if it exists
    logging.info('Model loaded from file.')
else:
    model = create_model()  # Create the model if it does not exist
    # Train the model
    model.fit(X_train_pad, y_train, epochs=30, batch_size=32, validation_split=0.2)
    model.save(model_file)  # Save the entire model
    logging.info('Model trained and saved to file.')

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test)
logging.info(f'Test Accuracy: {accuracy:.2f}')

def predict_anomaly(text):
    # Preprocess the input text
    text_seq = tokenizer.texts_to_sequences([text])  # Convert text to sequence
    text_pad = pad_sequences(text_seq, maxlen=max_length)  # Pad the sequence

    # Use the model to predict the score
    prediction = model.predict(text_pad)  # Get the prediction from the model

    # Convert prediction to a score (assuming binary classification)
    score = int(prediction[0][0] * 100)  # Scale the prediction to a score out of 100

    # Cap the score at 100
    if score > 100:
        score = 100

    logging.info(f'Predicted score for input "{text}": {score}')
    return score


def predict_anomaly(text):
    # Your existing code to predict the score
    score = 0  
    # Preprocess the input text
    text_seq = tokenizer.texts_to_sequences([text])  # Convert text to sequence
    text_pad = pad_sequences(text_seq, maxlen=max_length)  # Pad the sequence

    # Use the model to predict the score
    prediction = model.predict(text_pad)  # Get the prediction from the model

    # Convert prediction to a score (assuming binary classification)
    score = int(prediction[0][0] * 100)  # Scale the prediction to a score out of 100

    # Cap the score at 100
    if score > 100:
        score = 100

    logging.info(f'Predicted score for input "{text}": {score}')
    score = len(text) % 100
    return score

# Function to load model weights
def main():
    if len(sys.argv) > 1:
        # Call the predict_anomaly function with the input text
        getModelReadyForPrediction()
        input_text = sys.argv[1]
        score = predict_anomaly(input_text)
        print(score)  # Print only the score for the PHP script to capture

# Check if the script is being run directly
if __name__ == "__main__":
    main()
    

# New COde commented

'''

# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
# import sys
# import logging

# # Set up logging
# logging.basicConfig(filename='anomaly_detection.log', level=logging.INFO)

# # Load the dataset
# data = pd.read_csv('text_data.csv')  # Ensure this path is correct

# # Encode labels
# label_encoder = LabelEncoder()
# data['label'] = label_encoder.fit_transform(data['label'])

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# # Tokenize the text
# tokenizer = Tokenizer(num_words=5000)
# tokenizer.fit_on_texts(X_train)

# # Convert texts to sequences
# X_train_seq = tokenizer.texts_to_sequences(X_train)
# X_test_seq = tokenizer.texts_to_sequences(X_test)

# # Pad sequences to ensure uniform input size
# max_length = 100  # You can adjust this based on your data
# X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
# X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

# # Define the model
# def create_model():
#     model = Sequential()
#     model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_length))
#     model.add(LSTM(64, return_sequences=True))
#     model.add(Dropout(0.5))
#     model.add(LSTM(32))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='sigmoid'))  # Binary classification
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

# # Check if model already exists
# model_file = 'model.h5'
# if os.path.exists(model_file):
#     model = load_model(model_file)  # Load the entire model if it exists
#     logging.info('Model loaded from file.')
# else:
#     model = create_model()  # Create the model if it does not exist
#     # Train the model
#     model.fit(X_train_pad, y_train, epochs=30, batch_size=32, validation_split=0.2)
#     model.save(model_file)  # Save the entire model
#     logging.info('Model trained and saved to file.')

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test_pad, y_test)
# logging.info(f'Test Accuracy: {accuracy:.2f}')

# def predict_anomaly(text, applicant_data):
#     """
#     Predicts the anomaly score for the given text input and applicant data.
    
#     Parameters:
#     - text: The text input from the applicant (e.g., responses during verification).
#     - applicant_data: A dictionary containing additional applicant information (e.g., previous responses, application details).
    
#     Returns:
#     - score: An integer score indicating the likelihood of the application being fraudulent.
#     """
#     try:
#         # Preprocess the input text
#         text_seq = tokenizer.texts_to_sequences([text])  # Convert text to sequence
#         text_pad = pad_sequences(text_seq, maxlen=max_length)  # Pad the sequence

#         # Use the model to predict the score
#         prediction = model.predict(text_pad)  # Get the prediction from the model

#         # Log the raw prediction output
#         logging.info(f'Raw prediction output for input "{text}": {prediction}')

#         # Convert prediction to a score (assuming binary classification)
#         score = int(prediction[0][0] * 100)  # Scale the prediction to a score out of 100

#         # Cap the score at 100
#         if score > 100:
#             score = 100
#         elif score < 0:  # Ensure score is not negative
#             score = 0

#         # Analyze applicant behavior based on additional data
#         behavior_score = analyze_applicant_behavior(applicant_data)
#         final_score = (score + behavior_score) / 2  # Combine the scores for a final score

#         # Increase the final score based on certain conditions
#         if behavior_score > 0:
#             final_score += 10  # Increase score if there are minor flags

#         # Cap the final score at 100
#         if final_score > 100:
#             final_score = 100
#         elif final_score < 0:
#             final_score = 0

#         logging.info(f'Predicted score for input "{text}": {final_score}')
#         return final_score

#     except Exception as e:
#         logging.error(f'Error in predicting anomaly for input "{text}": {str(e)}')
#         return None  # Return None or an appropriate value in case of error

# def analyze_applicant_behavior(applicant_data):
#     """
#     Analyzes the applicant's behavior based on provided data to identify anomalies.
    
#     Parameters:
#     - applicant_data: A dictionary containing applicant information (e.g., previous responses, application details).
    
#     Returns:
#     - behavior_score: An integer score indicating the likelihood of fraudulent behavior based on applicant data.
#     """
#     behavior_score = 0

#     # Example checks for suspicious behavior
#     if applicant_data.get('previous_responses') != applicant_data.get('current_response'):
#         behavior_score += 20  # Flag inconsistent responses (increased weight)

#     if applicant_data.get('information_mismatch'):
#         behavior_score += 30  # Flag mismatched information (increased weight)

#     # Add more checks as necessary based on the specific criteria for fraud detection

#     # Ensure behavior score is capped at 100
#     if behavior_score > 100:
#         behavior_score = 100

#     return behavior_score

# # Function to load model weights
# def main():
#     if len(sys.argv) > 1:
#         input_text = sys.argv[1]
#         # Example applicant data for testing
#         applicant_data = {
#             'previous_responses': 'I live in New York.',
#             'current_response': input_text,
#             'information_mismatch': False  # This would be determined based on actual checks
#         }
        
#         score = predict_anomaly(input_text, applicant_data)
#         print(score)  # Output the score for the PHP script to capture

# # Check if the script is being run directly
# if __name__ == "__main__":
#     main()  # Run the main function
'''