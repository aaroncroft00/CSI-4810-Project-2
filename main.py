from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import spacy
import spacy.cli
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dropout, BatchNormalization
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras.utils as np_utils
from keras.optimizers import Adam



# # Setup Selenium WebDriver
# driver = webdriver.Chrome()  # Update this path
# url = "https://www.imdb.com/title/tt0816692/reviews?ref_=tt_ql_3"
# driver.get(url)
#
# # Specify the number of reviews you want to scrape
# desired_number_of_reviews = 5572  # Update this number as needed
#
# # Wait for the initial reviews to load
# time.sleep(2.5)
#
# # Click "Load More" button until the desired number of reviews is loaded
# current_review_count = 0
# while current_review_count < desired_number_of_reviews:
#     try:
#         load_more_button = WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.CLASS_NAME, "ipl-load-more__button"))
#         )
#         load_more_button.click()
#         # Wait for the new content to load
#         time.sleep(3)
#         current_review_count = len(driver.find_elements(By.CLASS_NAME, "review-container"))
#     except Exception as e:
#         print("All reviews loaded or an error occurred:", e)
#         break
#
# # Parse the page with BeautifulSoup
# soup = BeautifulSoup(driver.page_source, 'html.parser')
# driver.quit()
#
# # Extract reviews and ratings
# movie_containers = soup.find_all('div', class_='review-container')
# reviews = []
# ratings = []
#
# for container in movie_containers[:desired_number_of_reviews]:  # Limit to desired number of reviews
#     review = container.find('div', class_='text').get_text(strip=True)
#     reviews.append(review)
#
#     rating_tag = container.find('span', class_='rating-other-user-rating')
#     rating = rating_tag.get_text(strip=True) if rating_tag else 'No Rating'
#     ratings.append(rating)
#
# # Create DataFrame
# df_reviews = pd.DataFrame({'Review': reviews, 'Rating': ratings})
#
# # Replace 'No Rating' with NaN and then convert to float
# df_reviews['Rating'] = df_reviews['Rating'].replace('No Rating', np.nan)
# # Drop rows where 'Rating' is NaN
# df_reviews = df_reviews.dropna(subset=['Rating'])
# df_reviews['Rating'] = df_reviews['Rating'].str.split('/').str[0].astype(float)
# df_reviews.to_csv('C:/Users/aaron/Downloads/interstellar_reviews.csv', index=False)
#
# pd.set_option('display.max_colwidth', None)  # to display full review text
# ----------------------------------------------------------------------------------------------------------

df = pd.read_csv('interstellar_reviews.csv')

# plt.hist(df['Rating'], bins=10, range=(1, 10), edgecolor='black')
# plt.xlabel('Ratings')
# plt.ylabel('Number of Reviews')
# plt.title('Histogram of Review Ratings')
# plt.xticks(range(1, 11))  # Setting x-ticks to represent each rating class
# plt.show()
#
# text = " ".join(review for review in df.Review)
# # Add any additional stopwords if necessary
# stopWords = set(STOPWORDS)
# # Generate the word cloud
# wordcloud = WordCloud(
#     stopwords=stopWords,
#     background_color="white",
#     max_words=100,
#     width=500,
#     height=250,
#     colormap='twilight_shifted'  # you can change the colormap to others like 'plasma', 'magma'
# ).generate(text)
# # Display the generated word cloud:
# plt.figure(figsize=(17, 17))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()



# # Define the number of samples to keep
# n_samples_to_keep = 900  # Or any other number you've determined
#
# # Randomly select the samples to keep from the 10.0 class
# df_10_samples_to_keep = df[df.Rating == 10.0].sample(n=n_samples_to_keep, random_state=42)
#
# # Remove all 10.0 samples from the original DataFrame
# df = df[df.Rating != 10.0]
#
# # Add back only the randomly selected 10.0 samples
# df = pd.concat([df, df_10_samples_to_keep])
# # Check the class counts
# print(df.Rating.value_counts())


# df_majority = df[df.Rating == 10]  # Assuming class '10' is the majority class
# df_minority = df[df.Rating != 10]  # All other classes
# # Ensure that n_samples does not exceed the length of the majority class
# # Upsample minority class
# df_minority_upsampled = resample(df_minority,
#                                  replace=True,       # Sample with replacement
#                                  n_samples=len(df_majority),  # to match majority class
#                                  random_state=123)   # reproducible results
#
# # Combine majority class with upsampled minority class
# df_upsampled = pd.concat([df_majority, df_minority_upsampled])
#
# # Display new class counts
# print(df_upsampled.Rating.value_counts())
#
# # Downsample majority class
# df_majority_downsampled = resample(df_majority,
#                                    replace=False,     # Sample without replacement
#                                    n_samples=df_minority,  # Adjusted to not exceed the available samples
#                                    random_state=123)  # reproducible results
#
# # Combine minority class with downsampled majority class
# df_downsampled = pd.concat([df_minority, df_majority_downsampled])
#
# # Display new class counts
# print(df_downsampled.Rating.value_counts())

# # Download required NLTK resources if not already downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize the lemmatizer and stopwords list
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))


# def preprocess_text(text):
#     # Lowercase the text
#     text = text.lower()
#     # Remove punctuation
#     text = re.sub(r'[^\w\s]', '', text)
#     # Tokenization
#     tokens = nltk.word_tokenize(text)
#     # Remove stopwords and lemmatize
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
#     return ' '.join(tokens)


# Then load the model as usual
nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])


# def preprocess_text(text):
#     doc = nlp(text)
#     tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
#     return ' '.join(tokens)
#
#
# def document_vector(doc):
#     doc = nlp(doc)
#     vectors = [token.vector for token in doc if token.has_vector]
#
#     if len(vectors) == 0:
#         return np.zeros((nlp.vocab.vectors_length,))  # Return a zero vector if no words have vectors
#     else:
#         return np.mean(vectors, axis=0)
#
#
# # Apply the preprocessing to each review
# df['processed_review'] = df['Review'].apply(preprocess_text)
# X = np.array([document_vector(text) for text in df['processed_review']])
# vectorizer = TfidfVectorizer()

# X = vectorizer.fit_transform(df['processed_review'])

# Process texts in batches and return vectors
def texts_to_vectors(texts):
    # Preprocess and vectorize texts in a batch
    text_vectors = []
    for doc in nlp.pipe(texts, batch_size=20):  # Adjust batch_size as needed
        tokens = [token for token in doc if not token.is_stop and not token.is_punct and token.has_vector]
        if tokens:
            text_vectors.append(np.mean([token.vector for token in tokens], axis=0))
        else:
            text_vectors.append(np.zeros((nlp.vocab.vectors_length,)))
    return np.array(text_vectors)


X = texts_to_vectors(df['Review'])
# print(X)

# X = vectorizer.transform(df['processed_review'])
# Your target variable
y = df['Rating'].astype(float)


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)


X_padded = pad_sequences(X, maxlen=100, truncating='post', padding='post')
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Actual vs Predicted scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.show()
#
plt.figure(figsize=(14, 6))

# Histogram for actual labels
plt.subplot(1, 2, 1)
sns.histplot(y_test, bins=np.arange(11)-0.5, color='blue')
plt.title('Actual Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks(range(10))  # Adjust this range according to your rating scale

# Histogram for predicted labels
plt.subplot(1, 2, 2)
sns.histplot(y_pred, bins=np.arange(11)-0.5, color='orange')
plt.title('Predicted Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks(range(10))  # Adjust this range according to your rating scale

plt.tight_layout()
plt.show()



# def categorize_rating(rating):
#     if rating <= 4:
#         return 'low'
#     elif rating <= 7:
#         return 'medium'
#     else:
#         return 'high'
#
#
# df['RatingCategory'] = df['Rating'].apply(categorize_rating)

df['Rating'] = df['Rating'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=2000, multi_class='multinomial', class_weight='balanced')


# Train the model
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# # Actual vs Predicted scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.title('Actual vs Predicted Ratings')
# plt.xlabel('Actual Ratings')
# plt.ylabel('Predicted Ratings')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
# plt.show()

# Plotting histogram of the actual vs predicted
plt.hist([y_test, y_pred], color=['blue', 'red'], alpha=0.5, label=['Actual', 'Predicted'])
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title('Histogram of Actual vs Predicted Classes')
plt.legend()
plt.show()

# Adjust class labels to start from 0
y_train = y_train - 1
y_test = y_test - 1

# Initialize the XGBClassifier
xgb = XGBClassifier()

# Train the model
xgb.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb.predict(X_test)
# # Calculate metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plotting histogram of the actual vs predicted
plt.hist([y_test, y_pred], color=['blue', 'red'], alpha=0.5, label=['Actual', 'Predicted'])
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title('Histogram of Actual vs Predicted Classes')
plt.legend()
plt.show()


Initialize RandomForestClassifier with class_weight set to 'balanced'
rf_clf = RandomForestClassifier(
    n_estimators=50,  # Reduced number of trees
    max_depth=10,     # Limit the depth of each tree
    min_samples_split=4,  # Minimum number of samples required to split a node
    min_samples_leaf=4,   # Minimum number of samples required at each leaf node
    max_features='sqrt',  # Consider a subset of features for each split
    class_weight='balanced',  # Keep class_weight to address imbalances
    n_jobs=-1,  # Use all cores
    random_state=42
)

# Train the model
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = rf_clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Plotting histogram of the actual vs predicted
plt.hist([y_test, y_pred], color=['blue', 'red'], alpha=0.5, label=['Actual', 'Predicted'])
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title('Histogram of Actual vs Predicted Classes')
plt.legend()
plt.show()



Assume df_reviews['Review'] is your text data and df_reviews['Rating'] are your labels
Preprocessing: Tokenization and Padding
tokenizer = Tokenizer(num_words=5000)  # Adjust num_words as needed
tokenizer.fit_on_texts(df['Review'])
sequences = tokenizer.texts_to_sequences(df['Review'])
Xnew = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as per your data
ynew = pd.get_dummies(df['Rating']).values

sm = SMOTE(random_state=42, k_neighbors=1, sampling_strategy={0: 2000, 1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000,
                                                              6: 2000, 7: 2000, 8: 2000})
X_res, y_res = sm.fit_resample(Xnew, ynew)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=100))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

# Adding a dense layer
model.add(Dense(128, activation='relu'))

# Adding dropout for regularization
model.add(Dropout(0.5))

# Adding an additional dense layer
model.add(Dense(64, activation='relu'))

# Adding dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(10, activation='softmax'))  # 10 for the 10 rating classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=16, epochs=20, validation_split=0.1)
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

predictions = model.predict(X_test)
# If you need class labels instead of probabilities
predicted_classes = np.argmax(predictions, axis=1)

# To get the corresponding rating labels, you might need to inverse transform
# using the columns from pd.get_dummies if you have the mapping available
# rating_labels = dict(enumerate(df['Rating'].astype('category').cat.categories))
# predicted_ratings = [rating_labels[idx] for idx in predicted_classes]

# Now you can see the predicted ratings
# print(predicted_ratings)
# Convert one-hot encoded y_test back to class labels
actual_labels = np.argmax(y_test, axis=1)

# Convert softmax predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Plotting
plt.figure(figsize=(12, 6))

# Histogram for actual labels
plt.subplot(1, 2, 1)
sns.histplot(actual_labels, kde=False, bins=np.arange(11)-0.5, color='blue')
plt.title('Actual Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks(range(10))

# Histogram for predicted labels
plt.subplot(1, 2, 2)
sns.histplot(predicted_labels, kde=False, bins=np.arange(11)-0.5, color='orange')
plt.title('Predicted Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks(range(10))

plt.tight_layout()
plt.show()

# Define the RNN architecture
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))  # Adjust 'input_dim' and 'output_dim' as needed
model.add(LSTM(128, return_sequences=False))  # 64 LSTM units; 'return_sequences=False' because we only need the last output
model.add(Dense(10, activation='softmax'))  # Output layer; '10' for the 10 rating classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=4, validation_data=(X_res, y_res))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
# Make predict
predictions = model.predict(X_test)
# print(classification_report(y_test, predictions))

actual_labels = np.argmax(y_test, axis=1)

# Convert softmax predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Plotting
plt.figure(figsize=(12, 6))

# Histogram for actual labels
plt.subplot(1, 2, 1)
sns.histplot(actual_labels, kde=False, bins=np.arange(11)-0.5, color='blue')
plt.title('Actual Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks(range(10))

# Histogram for predicted labels
plt.subplot(1, 2, 2)
sns.histplot(predicted_labels, kde=False, bins=np.arange(11)-0.5, color='orange')
plt.title('Predicted Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks(range(10))

plt.tight_layout()
plt.show()
#
# # Step 2: Convert text data to numerical features (TF-IDF)
tfidf = TfidfVectorizer(max_features=3500)  # Adjust max_features as needed
X = tfidf.fit_transform(df['Review']).toarray()
# y = df['Rating']
#
# # Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the StandardScaler
scaler = MinMaxScaler()

# Fit on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to test data
X_test_scaled = scaler.transform(X_test)
# Step 4: Train the SVM classifier


svm_classifier = SVC(kernel='rbf', C=0.7, class_weight='balanced')
svm_classifier.fit(X_train, y_train)

# Step 5: Evaluate the classifier
y_pred = svm_classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

actual_labels = y_test

# Plotting histogram of the actual vs predicted
plt.hist([y_test, y_pred], color=['blue', 'red'], alpha=0.5, label=['Actual', 'Predicted'])
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title('Histogram of Actual vs Predicted Classes')
plt.legend()
plt.show()

# Convert softmax predictions to class labels
predicted_labels = np.argmax(y_pred, axis=1)

plt.figure(figsize=(14, 6))

# Histogram for actual labels
plt.subplot(1, 2, 1)
sns.histplot(actual_labels, bins=np.arange(11)-0.5, color='blue')
plt.title('Actual Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks(range(10))  # Adjust this range according to your rating scale

# Histogram for predicted labels
plt.subplot(1, 2, 2)
sns.histplot(y_pred, bins=np.arange(11)-0.5, color='orange')
plt.title('Predicted Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks(range(10))  # Adjust this range according to your rating scale

plt.tight_layout()
plt.show()
df['Review'] = pd.Series(df['Review'])


df['Processed_Review'] = df['Review'].str.lower()  # Lowercasing
# # Further cleaning can be done as required (like removing special chars, numbers, etc.)
#
# Tokenization and Vectorization
tokenizer = Tokenizer(num_words=5000)  # Adjust num_words as needed
tokenizer.fit_on_texts(df['Processed_Review'])
sequences = tokenizer.texts_to_sequences(df['Processed_Review'])
word_index = tokenizer.word_index

# Sequence Padding
data = pad_sequences(sequences, maxlen=500)  # Adjust maxlen as needed

# Convert ratings to categorical labels
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(df['Rating'])
dummy_y = np_utils.to_categorical(encoded_Y)
#
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, dummy_y, test_size=0.3, random_state=42)
# smote = SMOTE(k_neighbors=1)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Neural Network Architecture
model = Sequential()
model.add(Embedding(len(word_index) + 1, 128, input_length=500))  # Adjust embedding dimensions as needed
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(100))
model.add(Dense(10, activation='softmax'))  # 10 classes for the ratings 1-10

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Model summary
print(model.summary())

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')


# Get model predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print classification report
print(classification_report(y_true, y_pred_classes))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(conf_matrix)

# Plotting scatter plot of the actual vs predicted
plt.scatter(y_test, y_pred, c='blue', alpha=0.5, label='Actual vs Predicted')
plt.xlabel('Actual Classes')
plt.ylabel('Predicted Classes')
plt.title('Scatter Plot of Actual vs Predicted Classes')
plt.legend()
plt.show()

embeddings_index = {}
with open('glove.twitter.27B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
#
embedding_dim = 100  # This should match the GloVe embeddings you are using
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


from keras.callbacks import EarlyStopping

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)


model = Sequential()
model.add(Embedding(len(word_index) + 1, embedding_dim,
                    weights=[embedding_matrix],
                    input_length=100,
                    trainable=False))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(10, activation='softmax'))  # For 10 rating classes


initial_learning_rate = 0.05  # Set your initial learning rate here
optimizer = Adam(lr=initial_learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_smote, y_train_smote,
                    epochs=50,
                    batch_size=256,
                    callbacks=[early_stopping],
                    validation_data=(X_test, y_test))


loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print classification report for additional metrics
print(classification_report(y_true, y_pred_classes))

# Plotting histogram of the actual vs predicted
plt.hist([y_true, y_pred_classes], color=['blue', 'red'], alpha=0.5, label=['Actual', 'Predicted'])
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title('Histogram of Actual vs Predicted Classes')
plt.legend()
plt.show()

# Plotting scatter plot of the actual vs predicted
plt.scatter(y_true, y_pred_classes, c='blue', alpha=0.5, label='Actual vs Predicted')
plt.xlabel('Actual Classes')
plt.ylabel('Predicted Classes')
plt.title('Scatter Plot of Actual vs Predicted Classes')
plt.legend()
plt.show()

