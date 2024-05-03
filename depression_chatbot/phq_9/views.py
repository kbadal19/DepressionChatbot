from django.shortcuts import render


from rest_framework.views import APIView
from rest_framework.response import Response
from .models import PHQ9Question,PHQResponse
from rest_framework import status
from .serializers import PHQ9QuestionSerializer,PHQResponseSerializer
from django.db.models import Max
import joblib
class PHQ9QuestionList(APIView):
    def get(self, request):
        questions = PHQ9Question.objects.all()
        serializer = PHQ9QuestionSerializer(questions, many=True)
        return Response(serializer.data)


from django.shortcuts import get_object_or_404
from django.db.models import Max
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import PHQResponse
from django.contrib.auth.models import User
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
class PHQResponseCreate(APIView):
    def post(self, request):
        # Get the user ID from the request data
        user_id = request.data.get('user')
        
        # Retrieve the user instance
        user = get_object_or_404(User, pk=user_id)

        # Get the maximum batch number for the user's previous responses
        max_batch = PHQResponse.objects.filter(user=user).aggregate(Max('batch'))['batch__max']

        # Calculate the batch number
        batch_number = max_batch + 1 if max_batch is not None else 1
        
        voting_classifier = joblib.load(r'phq_9\voting_classifier_model.pkl')

        # Iterate over each response in the payload
        for response_data in request.data.get('responses', []):
            # Get question ID and response text from the payload
            question_id = response_data.get('question_id')
            response_text = response_data.get('response_text')
            predicted_emotion = voting_classifier.predict([response_text])[0]

            # Calculate depressive index based on predicted emotion
            depressive_index = get_depressive_index(predicted_emotion)

            # Rate the emotion based on depressive index
            sentiment_score = rate_emotion(depressive_index)

            # Retrieve the question instance
            question = get_object_or_404(PHQ9Question, pk=question_id)

            # Create the PHQResponse instance
            PHQResponse.objects.create(
                user=user,
                question=question,
                response_text=response_text,
                predicted_emotion=predicted_emotion,
                sentiment_score=sentiment_score,
                batch=batch_number
            )

        return Response({"message": "Responses created successfully"}, status=status.HTTP_201_CREATED)

def rate_emotion(depressive_index):
    if depressive_index >= 0.4:
        return 3  # Severely depressed
    elif 0.2 <= depressive_index < 0.4:
        return 2  # Moderately depressed
    elif 0.1 <= depressive_index < 0.2:
        return 1  # Mildly depressed
    else:
        return 0  # Not depressed

def get_depressive_index(emotion):
    # Define depressive index for each emotion
    depressive_index_mapping = {
        'Anger': 0.47,
        'Anticipation': -0.43,
        'Disgust': 0.17,
        'Fear': 0.45,
        'Joy': -0.56,
        'Sadness': 0.32,
        'Surprise': -0.28,
        'Trust': -0.19
    }
    return depressive_index_mapping.get(emotion, 0)  # Default to 0 if emotion not found

class PHQScore(APIView):
    def get(self, request, user_id):
        # Retrieve the user instance
        user = get_object_or_404(User, pk=user_id)

        # Call the get_score method
        score = self.get_score(user)

        return Response({"score": score}, status=status.HTTP_200_OK)

    def get_score(self, user):
        # Get the latest batch number for the user's responses
        latest_batch = PHQResponse.objects.filter(user=user).aggregate(Max('batch'))['batch__max']

        if latest_batch is not None:
            # Retrieve all responses of the latest batch
            latest_responses = PHQResponse.objects.filter(user=user, batch=latest_batch)

            # Calculate the sum of sentiment scores from the latest responses
            total_sentiment_score = sum(response.sentiment_score for response in latest_responses)

            return total_sentiment_score
        else:
            return 0


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import nltk

# # Download the punkt tokenizer from NLTK
# nltk.download('punkt')

# from nltk.tokenize import word_tokenize
# from tensorflow.keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Embedding

# from sklearn.preprocessing import LabelEncoder

# import warnings
# warnings.filterwarnings('ignore')

# # Set seaborn visualization style
# sns.set()

# # Read the training data from CSV
# train = pd.read_csv("train_sent_emo.csv")
# train.head()

# # Tokenize and preprocess the text data
# corpus = []

# for text in train['text']:
#     # Check if the text is a valid string, if not, skip it
#     if isinstance(text, str):
#         words = (word.lower() for word in word_tokenize(text))
#         corpus.append(list(words))

# # Count the number of unique words in the corpus
# num_words = len(corpus)

# # Split the data into training and testing sets
# train_size = int(train.shape[0] * 0.8)
# X_train = train.text[:train_size].astype(str)
# Y_train = train.sentiment[:train_size]

# X_test = train.text[train_size:].astype(str)
# Y_test = train.sentiment[train_size:]

# # Tokenize the text data using Keras Tokenizer
# tokenizer = Tokenizer(num_words)
# tokenizer.fit_on_texts(X_train)
# X_train = tokenizer.texts_to_sequences(X_train)
# X_train = pad_sequences(X_train, maxlen=32, truncating='post', padding='post')

# X_test = tokenizer.texts_to_sequences(X_test)
# X_test = pad_sequences(X_test, maxlen=32, truncating='post', padding='post')

# # Encode the target labels
# le = LabelEncoder()
# Y_train = le.fit_transform(Y_train)
# Y_test = le.transform(Y_test)

# # Create a Sequential model
# model = Sequential()
# model.add(Embedding(input_dim=num_words, output_dim=100, input_length=32, trainable=True))   # Add an Embedding layer
# model.add(LSTM(100, dropout=0.1, return_sequences=True))      # Add LSTM layers
# model.add(LSTM(100, dropout=0.1))
# model.add(Dense(1, activation='sigmoid'))

# # Compile the model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train the model
# history = model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_data=(X_test, Y_test))


# # Print model summary
# model.summary()

# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# # Initialize the VADER sentiment analyzer
# analyzer = SentimentIntensityAnalyzer()
# # Example sentence
# # sentence = "sometimes i do not have but mostly yess"
# sentence ="I find speaking slowly redundant...like who in the right mind would speak slowly just for people to recognise or pay attention to him"

# # Analyze the sentiment of the sentence
# sentiment_scores = analyzer.polarity_scores(sentence)
# print("overall score", sentiment_scores)

# Determine the sentiment category
# filtered_scores = {key: value for key, value in sentiment_scores.items() if key != 'compound'}
# max_score_label = max(filtered_scores, key=lambda key: filtered_scores[key])
# # Define the custom ranges for compound scores
# score_ranges = {
#     (float('-inf'), -0.25): 3,
#     (-0.25, 0): 2,
#     (0, 0.25): 1,
#     (0.25, float('inf')): 0
# }
# # Determine the sentiment score based on the 'compound' value
# compound_value = sentiment_scores['compound']
# for range_, score in score_ranges.items():
#     if range_[0] < compound_value < range_[1]:
#         sentiment_score = score

# print("Sentiment Score:", sentiment_score)

# import joblib
# # import pdb; pdb.set_trace()
# custom_text = "I find speaking slowly redundant...like who in the right mind would speak slowly just for people to recognise or pay attention to him"

# voting_classifier = joblib.load(r'phq_9\voting_classifier_model.pkl')

# predicted_emotion = voting_classifier.predict([custom_text])
# print("Predicted Emotion:", predicted_emotion[0])
