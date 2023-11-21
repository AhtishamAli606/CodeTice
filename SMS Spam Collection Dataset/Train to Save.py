import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin1')

# Preprocessing
X = data['v2']
y = data['v1'].map({'spam': 1, 'ham': 0})  # Convert 'spam' to 1 and 'ham' to 0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train and evaluate models

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_predictions = nb_model.predict(X_test_tfidf)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)
lr_predictions = lr_model.predict(X_test_tfidf)

# Support Vector Machine
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
svm_predictions = svm_model.predict(X_test_tfidf)


# Evaluate models
def evaluate_model(model, predictions, y_true):
    accuracy = accuracy_score(y_true, predictions)
    print(f"Model: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_true, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_true, predictions))
    print("\n")


# Evaluate Naive Bayes model
evaluate_model(nb_model, nb_predictions, y_test)

# Evaluate Logistic Regression model
evaluate_model(lr_model, lr_predictions, y_test)

# Evaluate Support Vector Machine model
evaluate_model(svm_model, svm_predictions, y_test)

# Save the trained models to files
joblib.dump(nb_model, 'nb_model.joblib')
joblib.dump(lr_model, 'lr_model.joblib')
joblib.dump(svm_model, 'svm_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

print("Models trained, tested, and saved successfully.")
