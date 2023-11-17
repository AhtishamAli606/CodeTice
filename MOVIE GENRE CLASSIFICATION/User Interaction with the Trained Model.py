from joblib import load

# Load the trained model and vectorizer
classifier = load('movie_genre_classifier.joblib')
tfidf_vectorizer = load('tfidf_vectorizer.joblib')

# Interactive prediction loop
while True:
    # Get user input
    user_input = input("Enter a movie description (or type 'exit' to end): ")

    if user_input.lower() == 'exit':
        break

    # Vectorize the user input
    user_input_tfidf = tfidf_vectorizer.transform([user_input])

    # Make a prediction
    prediction = classifier.predict(user_input_tfidf)

    # Display the predicted genre
    print(f"Predicted Genre: {prediction[0]}")
