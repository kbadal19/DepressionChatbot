
import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'emotions-dataset-for-nlp:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F605165%2F1085454%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240429%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240429T175728Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D02d2946ab798b7d52516cf16eb20afb365516af2f4133464348cf4265639ea2c3c1a127954041a130d4cbd0dc8d2b0f5ee5e4d4472b838f6257adc536b99c5c13ea715ea0382244b543a540dde23cf6de75a186eb372b571941211b632ecb9867ef3f81c68d8c105c498becfd93912d33858a8fbd8e2c2383864f0077c56e69ad8be18c7688f35bc30f95d42f5dd2839360710f2cfff04fa58beffd0eff3a40c76fe54a3b0a8879b59dfe61313a20f115541802a7e85ba0de52139e190ca3763c8fa0f294904268de1ed63fa61a94c2638c1bc03948f90cfc83cd2a8bc5e2b06154af65ac8289b983c4ecba293bc0a9576bb07c5aaebff5fd55e1f2409cc4db7'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

file_path = '/kaggle/input/emotions-dataset-for-nlp/val.txt'
val_df = pd.read_csv(file_path, sep=';', header=None, names=['Text', 'Emotion'])
file_path = '/kaggle/input/emotions-dataset-for-nlp/test.txt'
test_df = pd.read_csv(file_path, sep=';', header=None, names=['Text', 'Emotion'])
file_path = '/kaggle/input/emotions-dataset-for-nlp/train.txt'
train_df = pd.read_csv(file_path, sep=';', header=None, names=['Text', 'Emotion'])

train_df.info()
print('-----------------------------------------------------------------------')
test_df.info()
print('-----------------------------------------------------------------------')
val_df.info()

val_df['Emotion'].value_counts()

val_df['text_length'] = val_df['Text'].apply(len)
emo = val_df.iloc[val_df['text_length'].idxmax()][1]
txt = val_df.iloc[val_df['text_length'].idxmax()][0]
print('the text is: '+ txt)
print('the emotion is: '+ emo)

nltk.download('stopwords')
print(stopwords.words('english'))

def text_processing(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

val_df['Text'] = val_df['Text'].apply(text_processing)
train_df['Text'] = train_df['Text'].apply(text_processing)
test_df['Text'] = test_df['Text'].apply(text_processing)
train_df.sample(10)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
nb_classifier = make_pipeline(TfidfVectorizer(), MultinomialNB())
svm_classifier = make_pipeline(TfidfVectorizer(), SVC())
x = train_df['Text']
y = train_df['Emotion']
nb_classifier.fit(x, y)
svm_classifier.fit(x, y)

nb_predictions = nb_classifier.predict(test_df['Text'])
svm_predictions = svm_classifier.predict(test_df['Text'])
nb_accuracy = accuracy_score(test_df['Emotion'], nb_predictions)
svm_accuracy = accuracy_score(test_df['Emotion'], svm_predictions)

print("Naive Bayes Classifier Accuracy:", nb_accuracy)
print("SVM Classifier Accuracy:", svm_accuracy)

from sklearn.ensemble import RandomForestClassifier
rf_classifier = make_pipeline(TfidfVectorizer(), RandomForestClassifier(random_state=42))
rf_classifier.fit(x, y)

rf_predictions = rf_classifier.predict(test_df['Text'])
rf_accuracy = accuracy_score(test_df['Emotion'], rf_predictions)
print("Random Forest Classifier Accuracy:", rf_accuracy)

from sklearn.tree import DecisionTreeClassifier
dt_classifier = make_pipeline(TfidfVectorizer(), DecisionTreeClassifier(random_state=42))
dt_classifier.fit(train_df['Text'], train_df['Emotion'])

dt_predictions = dt_classifier.predict(test_df['Text'])
dt_accuracy = accuracy_score(test_df['Emotion'], dt_predictions)
print("Decision Tree Classifier Accuracy:", dt_accuracy)

from sklearn.neighbors import KNeighborsClassifier
knn_classifier = make_pipeline(TfidfVectorizer(), KNeighborsClassifier())
knn_classifier.fit(train_df['Text'], train_df['Emotion'])

knn_predictions = knn_classifier.predict(test_df['Text'])
knn_accuracy = accuracy_score(test_df['Emotion'], knn_predictions)
print("KNN Classifier Accuracy:", knn_accuracy)

from sklearn.linear_model import LogisticRegression
logistic_classifier = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
logistic_classifier.fit(train_df['Text'], train_df['Emotion'])

logistic_predictions = logistic_classifier.predict(test_df['Text'])
logistic_accuracy = accuracy_score(test_df['Emotion'], logistic_predictions)
print("Logistic Regression Classifier Accuracy:", logistic_accuracy)

from sklearn.ensemble import GradientBoostingClassifier
gbm_classifier = make_pipeline(TfidfVectorizer(), GradientBoostingClassifier())
gbm_classifier.fit(train_df['Text'], train_df['Emotion'])

gbm_predictions = gbm_classifier.predict(test_df['Text'])
gbm_accuracy = accuracy_score(test_df['Emotion'], gbm_predictions)
print("Gradient Boosting Classifier Accuracy:", gbm_accuracy)

accuracies = {
    "Naive Bayes": nb_accuracy,
    "SVM": svm_accuracy,
    "Random Forest": rf_accuracy,
    "Decision Tree": dt_accuracy,
    "KNN": knn_accuracy,
    "Logistic Regression": logistic_accuracy,
    "Gradient Boosting": gbm_accuracy
}

# Sort accuracies in descending order
sorted_accuracies = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)

# Extract model names and accuracies
model_names = [model[0] for model in sorted_accuracies]
accuracy_values = [model[1] for model in sorted_accuracies]

# Create countplot
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=model_names, y=accuracy_values, palette="viridis")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Accuracy of Models")
plt.xticks(rotation=45, ha='right')

# Annotate each bar with its accuracy value
for i, v in enumerate(accuracy_values):
    ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

from sklearn.ensemble import VotingClassifier

estimators = [
    ("Random Forest", rf_classifier),
    ("Logistic Regression", logistic_classifier),
    ("SVM", svm_classifier),
    ("Gradient Boosting", gbm_classifier),
    ("Decision Tree", dt_classifier)
]
voting_classifier = VotingClassifier(estimators, voting='hard')
voting_classifier.fit(train_df['Text'], train_df['Emotion'])
voting_predictions = voting_classifier.predict(test_df['Text'])
voting_accuracy = accuracy_score(test_df['Emotion'], voting_predictions)
print("Voting Classifier Accuracy:", round(voting_accuracy*100, 2), '%')

custom_text = "I'm feeling happy and excited today"
predicted_emotion = voting_classifier.predict([custom_text])
print("Predicted Emotion:", predicted_emotion[0])

custom_text = "I'm realy don't even know why this is done for me!"
predicted_emotion = voting_classifier.predict([custom_text])
print("Predicted Emotion:", predicted_emotion[0])

custom_text = "I feel overwhelmed with sorrow"
predicted_emotion = voting_classifier.predict([custom_text])
print("Predicted Emotion:", predicted_emotion[0])