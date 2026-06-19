# Paper Notes: Efficient Estimation of Word Representations in Vector Space

## 1. The Central Claim
The central claim of the paper is that high-quality word embeddings (turning words into continuous numbers) can be trained on massive text datasets using simple neural network architectures instead of expensive deep networks. 

The authors claim that by removing non-linear hidden layers, the model can scale to billions of words efficiently, and can take only a day to train upon all of them. This allows the vectors to learn surprisingly precise linear relationships—such that semantic ideas like gender and tense map perfectly to vector geometry (like the vectors: King-Man+Woman=Queen).

They also introduce the concept of words having multiple degrees of similarity, and how algebraic operations can be used to map different vectors to each other. It is discussed how time and accuracy depends on the dimensionality of the word vectors and on the amount of the training data.

## 2. Core Architecture
To prove this claim, the authors propose two distinct, shallow model variants:
* **Continuous Bag-of-Words (CBOW):** It predicts a single target word given its surrounding context words. The order of context words does not affect the prediction. So basically it means we have a bunch of words and then we try to predict the probability of the next word being 'something' using that bunch, similar to how the LLMs work in their generation of information.
* **Continuous Skip-gram:** Takes a single target word and tries to predict its surrounding context words within a specific sliding window size. In this method though, we have a target word and we try to generate a sentence that contains that word. Sliding window is a fixed set of words around the central words which we try to see and make the given word relevant to.

The network architecture itself takes away the layered neural network components:
1.  **An Input Layer:** Maps a sparse, single word index (Token ID).
2.  **An Embedding Layer:** Acts as a continuous weights matrix lookup table, projecting the word ID into a dense vector space of dimensions ($D$).
3.  **An Output Layer:** A linear layer followed by a Softmax function that assigns probability values across the entire vocabulary ($V$).

## 3. Dataset and Evaluation metrics
* **Dataset:** Evaluated using a large Google News dataset containing roughly 6 billion total tokens, filtered down to a vocabulary size of 1 million unique words.
* **Baselines:** Compared directly against older, computationally heavy architectures like the Feed-Forward Neural Network Language Model (NNLM) and Recurrent Neural Network Language Models (RNNLM).
* **Evaluation Metric:** A newly introduced Word Analogy Task consisting of:
    * **Syntactic Questions:** Structural relationships (example, amazing with amazingly as lucky with luckily).
    * **Semantic Questions:** General knowledge concepts (example, Athens is caplital of"Greece as Delhi is capital of India).
    * **Success Metric:** Accuracy percentage computed by identifying the closest vector using cosine similarity.
