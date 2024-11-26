# Document Retrieval and Question Answering System

This project is a document retrieval and question answering system that uses TF-IDF for document vectorization, FAISS for efficient similarity search, and OpenAI's GPT-4 for generating responses based on the retrieved documents.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher
- An OpenAI API key. You can get one by signing up at [OpenAI](https://beta.openai.com/signup/).

## Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required Python packages:

    ```bash
    pip install openai faiss-cpu numpy scikit-learn chardet
    ```

3. Set up your OpenAI API key as an environment variable:

    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

## Usage

1. Place your text documents in a directory named `documents`. Each document should be a `.txt` file.

3. Run the script:

    ```bash
    python RAG_FAISS_nokey.py
    ```

4. The script will read the documents, vectorize them using TF-IDF, build a FAISS index, and then use OpenAI's GPT-4 to generate responses to queries based on the retrieved documents.

## Code Explanation

### Import Libraries

```python
import os
import openai
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import chardet
```
 

