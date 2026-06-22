This repository contains a basic implementation of the Word2Vec model using PyTorch. The code trains word embeddings on a single sentence: the quick brown fox jumped over the lazy dog

## Prerequisites & Dependencies
To run this code, you need Python installed along with the **PyTorch** library.
You can install PyTorch via pip. 
Run the following command in your terminal:
pip install torch

How to Run the Code
-------------------

1.  Copy the Python code provided in the script.
2.  Save it to a file named word2vec.py.    
3.  Open your terminal or command prompt, navigate to the folder where you saved the file, and run: python word2vec.py
    

Technical Terms Explained Simply
--------------------------------

*   **Tokenization:** Computers cannot read continuous sentences like humans. Tokenization is the process of breaking a raw string of text into individual pieces, or tokens(lowercase words).
    
*   **Vocabulary:** A unique list of all the distinct words found in the text, sorted by how often they appear. The vocab size is just the total number of unique words.
    
*   **Word Embeddings:** It is similar to giving every word its own multi-dimensional vector system. A word embedding is a vector that represents a word. Words that share similar contexts will slowly drift closer together in this mathematical space during training.
    
*   **Embedding Dimension:** The number of coordinates we use to describe a single word. Here, every word is represented by a list of 10 numbers.
    
*   **Skip-Gram Architecture:** A strategy where we feed the model a single target word and ask it to guess the words next to it.
    
*   **Epoch:** One complete pass where the model looks at every single word-context pair in our dataset exactly once.
    

**What the Code Does**
----------------------

**Prepares the Text:** It takes the dataset and breaks it down into individual lowercase tokens and builds a small vocabulary.

*   **Creates Training Pairs:** It uses a sliding window of 2 words. For every word, it pairs it up with its neighboring words.
    
*   **Trains the Model:** It passes the center words into a neural network to predict the surrounding context words. Over 10 epochs, the network learns to group related words closer together in a 10-dimensional space.
    
*   **Outputs the Vectors:** Finally, it gives out the learned embedding vectors for a few sample words and measures their mathematical length.
    

**What to Expect**
------------------

It is expected that when we run the script, we should see the training loss decrease every 5 epochs, followed by the final word vector statistics.
