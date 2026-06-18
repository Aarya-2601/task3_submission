Before writing a single line of code:
Read the abstract, introduction, and method section carefully
Identify the central claim - what is the paper actually asserting works better, and why?
Identify the core architecture or algorithm - what exactly needs to be implemented to test that claim?
Note the dataset, evaluation metric, and the baseline they compare against
Write a short PAPER_NOTES.md (half a page is fine) answering these three questions before you start coding. This is submitted alongside your code.

# Paper Notes: Efficient Estimation of Word Representations in Vector Space

## 1. The Central Claim
The central claim of the paper is that high-quality word embeddings (turning words into continuous numbers) can be trained on massive text datasets using simple, shallow, log-linear neural network architectures instead of deep, expensive networks. 

The authors assert that by removing non-linear hidden layers, the model can scale to billions of words efficiently. This allows the vectors to learn surprisingly precise linear relationships—such that semantic ideas like gender and tense map perfectly to vector geometry (e.g., the famous vector relationship: $$\text{Vector}(\text{"King"}) - \text{Vector}(\text{"Man"}) + \text{Vector}(\text{"Woman"}) \approx \text{Vector}(\text{"Queen"})$$).

## 2. Core Architecture / Algorithm
To prove this claim, the authors propose two distinct, shallow model variants:
* **Continuous Bag-of-Words (CBOW):** Predicts a single target word given its surrounding context words. The order of context words does not affect the prediction.
* **Continuous Skip-gram:** Takes a single target word and tries to predict its surrounding context words within a specific sliding window size.

The network architecture itself strips away the heavy multi-layer neural network components:
1.  **An Input Layer:** Maps a sparse, single word index (Token ID).
2.  **An Embedding Layer:** Acts as a continuous weights matrix lookup table, projecting the word ID into a dense vector space of dimensions ($D$).
3.  **An Output Layer:** A linear layer followed by a Softmax function that assigns probability values across the entire vocabulary ($V$).

## 3. Dataset, Baseline, and Evaluation Metrics
* **Dataset:** Evaluated using a large Google News dataset containing roughly 6 billion total tokens, filtered down to a vocabulary size of 1 million unique words.
* **Baselines:** Compared directly against older, computationally heavy architectures like the Feed-Forward Neural Network Language Model (NNLM) and Recurrent Neural Network Language Models (RNNLM).
* **Evaluation Metric:** A newly introduced Word Analogy Task consisting of:
    * **Syntactic Questions:** Structural relationships (e.g., "amazing" is to "amazingly" as "lucky" is to "luckily").
    * **Semantic Questions:** General knowledge concepts (e.g., "Athens" is to "Greece" as "Berlin" is to "Germany").
    * **Success Metric:** Accuracy percentage (%) computed by identifying the closest vector using cosine similarity.
