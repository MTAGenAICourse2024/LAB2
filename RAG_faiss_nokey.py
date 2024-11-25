import os
import openai
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import chardet

#If needed get rid of  proxy
#del os.environ["http_proxy"]
#del os.environ["https_proxy"]

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
#openai.api_key = "ENTER_YOUR_OPENAI_KEY"

# Directory containing the documents
documents_dir = "documents"



# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']


# Read documents from the directory
documents = []
filenames = []
for filename in os.listdir(documents_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(documents_dir, filename)
        encoding = detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
            documents.append(file.read())
            filenames.append(filename)

# Step 1: Vectorize the documents using TF-IDF
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents).toarray()

# Step 2: Build a FAISS index for efficient similarity search
index = faiss.IndexFlatL2(doc_vectors.shape[1])
index.add(doc_vectors)


def retrieve_relevant_docs(query, k=2):
    query_vector = vectorizer.transform([query]).toarray()
    distances, indices = index.search(query_vector, k)
    return [documents[i] for i in indices[0]]


def generate_response(query):
    # Retrieve relevant documents
    relevant_docs = retrieve_relevant_docs(query)
    context = " ".join(relevant_docs)

    # Generate response using OpenAI GPT-4 chat model
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}
        ],
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].message['content'].strip()


# Example query
#query = "Which languages does Joe Doe speak ?"
#query = "What is Jane Smith's educational background?"
#query = "What patents does Jane Smith hold?"
#query = "What awards has Jane Smith received?
#query = "What is Jane Smith's current role and responsibilities?
#query = "Who is oldest employee ?"
#query = "List the birth dates of all the employees"
#query = "So how come David Wilson the oldest ?"
query = "List the dates when all employees joined the company"

response = generate_response(query)
print(response)