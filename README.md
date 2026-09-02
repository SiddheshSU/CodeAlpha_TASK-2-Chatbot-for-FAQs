# FAQ Chatbot

## 📌 Project Overview

The FAQ Chatbot is a Python-based web application developed using Streamlit. It helps users get answers to frequently asked questions related to college services and student information.

The chatbot uses Natural Language Processing (NLP) techniques to preprocess questions and TF-IDF with cosine similarity to find the most relevant FAQ answer.

## ✨ Features

- Simple and user-friendly chatbot interface
- Collection of frequently asked questions
- Text preprocessing using NLP
- TF-IDF vectorization
- Cosine similarity for question matching
- Displays the most relevant answer
- Chat history support using Streamlit

## 🛠️ Technologies Used

- Python
- Streamlit
- NLTK
- Scikit-learn
- TF-IDF
- Cosine Similarity

## 🧠 How It Works

1. The user enters a question.
2. The question is cleaned and tokenized using NLP techniques.
3. TF-IDF converts the FAQ questions into numerical vectors.
4. Cosine similarity compares the user's question with the FAQ questions.
5. The chatbot selects the most similar question.
6. The corresponding answer is displayed to the user.

## ▶️ How to Run

Install the required libraries:

```bash
pip install streamlit nltk scikit-learn
