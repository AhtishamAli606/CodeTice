import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# Load your dataset
data = pd.read_csv('test_data_solution.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['DESCRIPTION'])

# Choose a classifier (Logistic Regression in this example)
classifier = LogisticRegression()
classifier.fit(X_train_tfidf, train_data['GENRE'])

# Save the trained model
dump(classifier, 'movie_genre_classifier.joblib')
dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

# Evaluate the model on the test set
X_test_tfidf = tfidf_vectorizer.transform(test_data['DESCRIPTION'])
predictions = classifier.predict(X_test_tfidf)

# Display results
accuracy = accuracy_score(test_data['GENRE'], predictions)
print(f'Training and Testing Results:')
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(test_data['GENRE'], predictions))
