import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configure NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
try:
    nltk.download('punkt')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Load intents from the JSON file
file_path = os.path.join(os.path.dirname(__file__), "intents.json")
try:
    with open(file_path, "r") as file:
        intents = json.load(file)
except (FileNotFoundError, json.JSONDecodeError) as e:
    st.error(f"Error loading intents file: {e}")
    st.stop()

# Validate intents structure
for intent in intents:
    if not all(key in intent for key in ['tag', 'patterns', 'responses']):
        st.error("Invalid structure in intents.json. Ensure 'tag', 'patterns', and 'responses' are included.")
        st.stop()

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=30000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        if pattern.strip():  # Avoid empty patterns
            tags.append(intent['tag'])
            patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function
def chatbot(input_text):
    try:
        input_text = vectorizer.transform([input_text])
        tag = clf.predict(input_text)[0]
        for intent in intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    except Exception as e:
        return f"Error processing input: {e}"

# Main application
counter = 0

def main():
    global counter
    st.title("Intents of Chatbot using NLP")

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message for hospital data and press Enter to start the conversation.")

        # Ensure chat log file exists
        if not os.path.exists('chat_log.csv'):
            try:
                with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
            except Exception as e:
                st.error(f"Error creating chat log: {e}")
                return

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Save to chat log
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([user_input, response, timestamp])
            except Exception as e:
                st.error(f"Error saving chat log: {e}")

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                return

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        except FileNotFoundError:
            st.write("No conversation history found.")

    # About Menu
    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

        st.subheader("Project Overview:")

        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
        """)

        st.subheader("Dataset:")

        st.write("""
        The dataset used in this project is hospital data intents and entities. The data is stored in a list.
        - Intents: The intent of the user input (e.g. "greeting", "appointment", "billing",etc)
        - Entities: The entities extracted from user input (e.g. "Hi", "What is my current bill??", "How can I check my bill online?")
        - Text: The user input text.
        """)

        st.subheader("Streamlit Chatbot Interface:")

        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("Conclusion:")

        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")


if __name__ == '__main__':
    main()
