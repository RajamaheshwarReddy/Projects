#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data=pd.read_csv(r"C:\Users\LENOVO\Downloads\search_engine\output.csv")


# In[2]:


data.head()


# In[3]:


ids=data['subtitle_id']


# In[4]:


subtitle_name=data['subtitle_name']


# In[5]:


ids


# In[6]:


content =data['subtitle_content']


# In[8]:


content_t=data['subtitle_name']


# In[7]:


content


# In[8]:


from transformers import BertTokenizer, BertModel
import numpy as np
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def preprocess_and_encode_batch(texts):
    
    texts = [text[:1000] for text in texts]

    tokens = tokenizer.batch_encode_plus(texts, add_special_tokens=True, return_tensors='pt', padding=True, truncation=True, max_length=1000)

    with torch.no_grad():
       
        outputs = bert_model(**tokens)

    bert_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return bert_embeddings

def chunk_data(data, chunk_size):
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    return chunks

chunk_size = 32

chunks = chunk_data(data['subtitle_content'], chunk_size)

encoded_embeddings = []

for chunk in chunks:
    
    chunk_embeddings = preprocess_and_encode_batch(chunk)

    
    encoded_embeddings.append(chunk_embeddings)


final_encoded_embeddings = np.vstack(encoded_embeddings)



# In[16]:


final_encoded_embeddings


# In[6]:


from transformers import BertTokenizer, BertModel
import numpy as np
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def preprocess_and_encode_batch(texts):
    
    texts = [text[:1000] for text in texts]

    
    tokens = tokenizer.batch_encode_plus(texts, add_special_tokens=True, return_tensors='pt', padding=True, truncation=True, max_length=1000)

    with torch.no_grad():
        
        outputs = bert_model(**tokens)

    
    bert_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return bert_embeddings


title_embedings = preprocess_and_encode_batch(data['subtitle_name'])


# In[19]:


item = 'Broker'


# In[20]:


embad=preprocess_and_encode_batch(item)


# In[32]:


embad[0]


# In[22]:


title_embedings[0]


# In[18]:


title_embedings.shape


# In[19]:


final_encoded_embeddings.shape


# In[20]:


final_encoded = np.concatenate((title_embedings, final_encoded_embeddings), axis=1)


# In[21]:


final_encoded.shape


# In[37]:


import chromadb
client = chromadb.PersistentClient(path="Search Engine db")


# In[39]:


client.list_collections()


# In[38]:


collection = client.create_collection(name="final_encoded_embeddings")


# In[11]:


subtitle_id_list = data['subtitle_id'].astype(str).tolist()


content = data['subtitle_content'].astype(str).tolist()


# In[9]:


content_t=data['subtitle_name'].astype(str).tolist()


# In[10]:


len(content_t)


# In[32]:


len(subtitle_id_list)


# In[33]:


len(content)


# In[34]:


len(final_encoded_embeddings)


# In[40]:


collection.add(
    embeddings=final_encoded_embeddings,
    documents=content,
   
    ids=subtitle_id_list
)


# In[44]:


collection.peek()


# In[57]:


title_embedings


# In[14]:


import chromadb
client=chromadb.PersistentClient(path=r"C:\Users\LENOVO\Downloads\search_engine\Search Engine db")


# In[15]:


collection=client.get_collection(name="final_encoded_embeddings")


# In[26]:


def generate_unique_ids(count):
    unique_ids = []
    current_id = 1
    while len(unique_ids) < count:
        unique_ids.append(str(current_id))
        current_id += 1
    return unique_ids

# Generate 4100 unique numerical IDs as strings
ids = generate_unique_ids(4100)

# Now you can pass the IDs to the function that expects string IDs
collection.add(
    embeddings=title_embedings,
    documents=content_t,
    ids=ids
)


# In[27]:


collection.peek()


# In[ ]:




