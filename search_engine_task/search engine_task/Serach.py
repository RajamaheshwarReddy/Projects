import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import chromadb


def generate_bert_embeddings(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

 
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()

    return embeddings

def main():
   
    st.set_page_config(
        page_title="Bert Based Movie Title Search",
        page_icon="🎬",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.markdown(
        """
        <style>
        /* Add custom CSS here */
        .title {
            color: #FF5733; /* Change title color */
            font-size: 36px; /* Change title font size */
            font-weight: bold; /* Make title bold */
            text-align: center; /* Center align title */
        }

        .search-input {
            border: 2px solid #FF5733; /* Add border to input field */
            border-radius: 5px; /* Add border radius to input field */
            padding: 8px; /* Add padding to input field */
            font-size: 16px; /* Change input font size */
            width: 70%; /* Set input width */
            margin: auto; /* Center align input field */
            display: block; /* Display input as block element */
            margin-bottom: 20px; /* Add bottom margin to input field */
        }

        .search-button {
            background-color: #FF5733; /* Change button background color */
            color: white; /* Change button text color */
            border: none; /* Remove button border */
            padding: 10px 20px; /* Add padding to button */
            font-size: 16px; /* Change button font size */
            border-radius: 5px; /* Add border radius to button */
            cursor: pointer; /* Add cursor pointer to button */
        }

        .search-button:hover {
            background-color: #FF814A; /* Change button background color on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

  
    st.markdown("<h1 class='title'>🔍 Bert Based Movie Title Search 🎬</h1>", unsafe_allow_html=True)

    
    user_input = st.text_input("Enter your movie title:", key="user_input")

    if st.button("Search 🔎"):
        if user_input:
           
            user_embeddings = generate_bert_embeddings(user_input)
            
            
            user_embeddings_list = user_embeddings.tolist()

           
            client = chromadb.PersistentClient(path=r"C:\Users\LENOVO\Downloads\search_engine\Search Engine db")
            collection = client.get_collection(name="final_encoded_embeddings")
            results = collection.query(query_embeddings=user_embeddings_list, include=["documents"])
            st.subheader("🔍 Search Results:")
            
            doc_count = 1
            for docs_list in results.get("documents", []):
                for doc in docs_list:
                    st.write(f"**Document {doc_count}:**")
                    st.write(doc.strip())
                    st.write("---")
                    doc_count += 1

if __name__ == "__main__":
    main()
