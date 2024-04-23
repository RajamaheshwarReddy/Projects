#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
import zipfile
import io

def extract_subtitle(db_file, batch_size=100, percentage=30):
    # Connect to the database
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    # Fetch total number of rows
    cur.execute("SELECT COUNT(*) FROM zipfiles")
    total_rows = cur.fetchone()[0]
    
    # Calculate number of batches for desired percentage
    total_batches = (total_rows * percentage) // (100 * batch_size)

    # Fetch data from the 'zipfiles' table in batches
    cur.execute("SELECT num, name, content FROM zipfiles LIMIT ?", (total_batches * batch_size,))
    
    data = []  # List to store subtitle data

    for _ in range(total_batches):
        batch = cur.fetchmany(batch_size)

        # Process each batch
        for row in batch:
            subtitle_id, subtitle_name, content = row
            subtitle_content = None

            # Check if the content is a ZIP archive
            if content.startswith(b'PK\x03\x04'):
                # Extract the content from the ZIP archive
                with zipfile.ZipFile(io.BytesIO(content)) as zip_file:
                    # Assume there is only one file in the archive
                    file_name = zip_file.namelist()[0]
                    subtitle_content = zip_file.read(file_name).decode('latin-1')

            else:
                # If not a ZIP archive, decode as Latin-1
                subtitle_content = content.decode('latin-1')

            # Append subtitle data to the list
            data.append({
                'subtitle_id': subtitle_id,
                'subtitle_name': subtitle_name,
                'subtitle_content': subtitle_content  # Store decoded content
            })

    # Close cursor and connection
    cur.close()
    conn.close()

    return data

# Example usage
db_file = r"C:\Users\LENOVO\Downloads\eng_subtitles_database.db"
data = extract_subtitle(db_file, percentage=5)


# In[2]:


data


# In[3]:


import pandas as pd
df = pd.DataFrame(data)


# In[4]:


df


# In[5]:


import re
def remove_words_in_brackets(text):
    return re.sub(r'\s*\([^)]*\)', '', text)

df['subtitle_content'] = df['subtitle_content'].apply(lambda x: remove_words_in_brackets(x))


# In[6]:


def remove_punctuation_and_numbers(text):
    
    text_without_punctuation = re.sub(r'[^\w\s]', '', text)
 
    text_without_numbers = re.sub(r'\d+', '', text_without_punctuation)
    return text_without_numbers

df['subtitle_content'] = df['subtitle_content'].apply(lambda x: remove_punctuation_and_numbers(x))


# In[7]:


df.head()


# In[8]:


def remove_symbols_and_numbers(text):
    return re.sub(r'[^A-Za-z\s]', '', text)
df['subtitle_content'] = df['subtitle_content'].apply(lambda x: remove_symbols_and_numbers(x))


# In[9]:


def remove_symbols_numbers_and_newlines(text):
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove symbols and numbers
    cleaned_text = re.sub(r'\r\n', '', cleaned_text)  # Remove \r\n characters
    return cleaned_text

df['subtitle_content'] = df['subtitle_content'].apply(lambda x: remove_symbols_numbers_and_newlines(x))


# In[10]:


df.subtitle_content[0]


# In[11]:


df['subtitle_content'] = df['subtitle_content'].str.replace('Watch any video online with OpenSUBTITLESFree Browser extension osdblinkext', '')


# In[12]:


df.head()


# In[13]:


df['subtitle_name'] = df['subtitle_name'].str.replace('.eng.1cd', '')


# In[14]:


df.head()


# In[15]:


df['subtitle_name'] = df['subtitle_name'].str.replace(r'(?<=\w)\.(?=\w)', ' ')


# In[16]:


df.subtitle_name[0]


# In[17]:


df['subtitle_name'] = df['subtitle_name'].apply(lambda x: x.replace('.', ' ').replace('(', '').replace(')', ''))



# In[18]:


df.subtitle_content[2]


# In[21]:


df.head()


# In[23]:


df.to_csv(r'C:\Users\LENOVO\Downloads\search_engine/output.csv', index=False)


# In[ ]:




