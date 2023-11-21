import joblib

# Load the trained models from files
nb_model = joblib.load('nb_model.joblib')
lr_model = joblib.load('lr_model.joblib')
svm_model = joblib.load('svm_model.joblib')

# Load the TF-IDF vectorizer used in training
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')


# Function to predict whether an SMS is spam or ham
def predict_sms(model, vectorizer, sms):
    sms_tfidf = vectorizer.transform([sms])
    prediction = model.predict(sms_tfidf)
    return "spam" if prediction[0] == 1 else "ham"


# Generate output until the user exits
while True:
    user_input = input("Enter an SMS (type 'exit' to end): ")

    if user_input.lower() == 'exit':
        break

    # Use one of the models for prediction (you can choose the model based on your preference)
    predicted_class = predict_sms(svm_model, tfidf_vectorizer, user_input)

    print(f"Predicted class: {predicted_class}\n")
