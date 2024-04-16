import streamlit as st
from openai import OpenAI

f = open('newkey_openai.txt')
OPENAI_API_KEY = f.read()

st.title("👀AI Code Reviewer")
st.subheader("Finds bugs in code and write the corrrected code")

client = OpenAI(api_key = OPENAI_API_KEY)

prompt = st.text_input("Enter a Code")

if st.button("Generate") == True:
    response = client.chat.completions.create(
      model="gpt-3.5-turbo-0613",
      messages=[
        {"role": "system", "content": "You are an Expert in code review. So, find bugs, errors and give the corrected code."},
        {"role": "user", "content": prompt}
      ]
    )
    st.write(response.choices[0].message.content)