import google.generativeai as genai
import streamlit as st

# Read Gemini API key from file
f = open('keys/gemini_API_key.txt')
key = f.read()

# Configure Gemini API
genai.configure(api_key=key)

# Set Streamlit app title and subheader
st.title("🤖 Data Science Tutor")
# st.subheader("AI Conversational Tutor")

# A note about what the app does
st.markdown("""
This application serves as an **'AI Conversational Tutor'** specifically developed to aid with queries regarding data science. Just enter your question in the designated area, and the AI Tutor will provide a response using its extensive knowledge.
""")

# Initialize Gemini model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    system_instruction="You are a helpful AI Teaching Assistant. Politely provide answers to user queries related to data science topics and provide helpful insights. If the query is unrelated, respond with 'I'm sorry, I'm not able to assist with that topic at the moment. However, if you have any other questions or need help with data science-related topics, feel free to ask!' If the user says 'hi', respond with 'Hi! How can I help you?'"
)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Dropdown menu for topic selection
topic_options = ["Data Science", "Machine Learning", "Data Visualization", "Statistics"]
selected_topic = st.selectbox("Select a topic:", topic_options)


# Start chat with model
chat = model.start_chat(history=st.session_state['chat_history'])

# Display chat history
for msg in chat.history:
    st.chat_message(msg.role).write(msg.parts[0].text)

# Get user input
user_prompt = st.chat_input()

# Process user input
if user_prompt:
    st.chat_message("user").write(user_prompt)
    response = chat.send_message(user_prompt)
    st.chat_message("ai").write(response.text)
    st.session_state["chat_history"] = chat.history