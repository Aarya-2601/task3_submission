import os
import urllib.request
import zipfile
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim

#DATASET DOWNLOADING AND PREPROCESSING
def load_text8_data():
    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = "text8.zip"
    extract_path = "text8"

    #Download Text8 if it doesn't exist
    if not os.path.exists(zip_path):
        print("Downloading Text8 dataset (approx. 31MB)...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")

    #Extract the zip file
    if not os.path.exists(extract_path):
        print("Extracting Text8 archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        print("Extraction complete.")

    # Read the text file
    with open(extract_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    #Use a sizeable sample
    return text[:500000]

print("Initializing data pipelines...")
raw_text = load_text8_data()
#Clean text to ensure only lowercase words remain
words = re.findall(r'[a-z]+', raw_text.lower())

#Build the Vocabulary
word_counts = Counter(words)
# Keeping words that appear at least 3 times ensures stable learning
vocab = [word for word, count in word_counts.items() if count >= 3]

#Inject validation anchor words if not present in the sample
for anchor in ["king", "man", "woman", "queen", "apple", "fruit"]:
    if anchor not in vocab:
        vocab.append(anchor)

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
v_size = len(vocab)
print(f"Dataset loaded. Total words in sample: {len(words)}. Cleaned Vocabulary Size: {v_size}")


#GENERATE SYMMETRICAL SLIDING WINDOW TRAINING PAIRS
def generate_pairs(words_list, w_to_i, window_size=2):
    training_pairs = []
    for i, target_word in enumerate(words_list):
        if target_word not in w_to_i:
            continue
        target_idx = w_to_i[target_word]
        
        start = max(0, i - window_size)
        end = min(len(words_list), i + window_size + 1)
        
        for j in range(start, end):
            if i != j and words_list[j] in w_to_i:
                context_idx = w_to_i[words_list[j]]
                training_pairs.append((target_idx, context_idx))
    return training_pairs

print("Generating training context pairs...")
pairs = generate_pairs(words, word_to_idx, window_size=2)
print(f"Total training pairs extracted from window: {len(pairs)}")


#DEFINE CORE LOG-LINEAR SKIP-GRAM MODEL
class SkipGramModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim=50):
        super(SkipGramModel, self).__init__()
        # Removes hidden layer completely, matching Mikolov's architecture optimization
        self.embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocabulary_size, bias=False)
        
    def forward(self, target_idx):
        # Shape: [batch_size, embedding_dim]
        embedded_representations = self.embeddings(target_idx)
        # Shape: [batch_size, vocabulary_size]
        output_logits = self.linear(embedded_representations)
        return output_logits


#TRAINING ENGINE SETUP
#Set configuration parameters
EMBEDDING_DIM = 50 
BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 0.005

#Check if hardware is available
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Training will execute on hardware device: {device}")

model = SkipGramModel(vocabulary_size=v_size, embedding_dim=EMBEDDING_DIM).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#Create batched training sets
targets_tensor = torch.tensor([p[0] for p in pairs], dtype=torch.long)
contexts_tensor = torch.tensor([p[1] for p in pairs], dtype=torch.long)
dataset = torch.utils.data.TensorDataset(targets_tensor, contexts_tensor)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


#MATHEMATICAL COVERSATION VALIDATION METRIC
def run_evaluation_check(trained_model, w_to_i):
    trained_model.eval()
    embeddings = trained_model.embeddings.weight.data.cpu()
    
    def get_cosine_similarity(v1, v2):
        return torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

    print("\n" + "="*50)
    print("      RESEARCH EVALUATION VERIFICATION METRICS      ")
    print("="*50)

    #Semantic Mapping
    if all(w in w_to_i for w in ["king", "man", "woman", "queen"]):
        v_king, v_man, v_woman, v_queen = embeddings[w_to_i["king"]], embeddings[w_to_i["man"]], embeddings[w_to_i["woman"]], embeddings[w_to_i["queen"]]
        target_analogy_vector = v_king - v_man + v_woman
        sim = get_cosine_similarity(target_analogy_vector, v_queen)
        print(f"↳ Cosine Similarity (king - man + woman || queen): {sim:.4f}")
    
    #Standard Semantic Category Grouping
    if all(w in w_to_i for w in ["apple", "fruit"]):
        v_apple, v_fruit = embeddings[w_to_i["apple"]], embeddings[w_to_i["fruit"]]
        sim_cat = get_cosine_similarity(v_apple, v_fruit)
        print(f"↳ Cosine Similarity (apple || fruit)             : {sim_cat:.4f}")
    print("="*50 + "\n")


#RUN EXECUTION LOOP
print("Starting training loops...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for batch_targets, batch_contexts in data_loader:
        batch_targets, batch_contexts = batch_targets.to(device), batch_contexts.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_targets)
        loss = criterion(logits, batch_contexts)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_targets.size(0)
        
    epoch_loss = running_loss / len(pairs)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Average Cross-Entropy Loss: {epoch_loss:.4f}")
    
    #Run evaluation at milestones to track structural learning
    if (epoch + 1) % 5 == 0 or epoch == 0:
        run_evaluation_check(model, word_to_idx)

print("Training finished successfully.")

def find_most_similar(word, trained_model, w_to_i, i_to_w, top_n=5):
    if word not in w_to_i:
        print(f"Word '{word}' is not in the vocabulary.")
        return

    trained_model.eval()
    embeddings = trained_model.embeddings.weight.data
    
    #Get the vector for our target word
    target_vec = embeddings[w_to_i[word]].unsqueeze(0) # Shape: [1, emb_dim]
    
    #Calculate cosine similarity between target_vec and ALL words in the vocabulary matrix
    #PyTorch does this rapidly using matrix operations
    similarities = torch.nn.functional.cosine_similarity(target_vec, embeddings, dim=1)
    
    #Get indices of the top highest similarity scores
    top_indices = torch.topk(similarities, top_n + 1).indices.tolist()
    
    print(f"\nWords most similar to '{word}':")
    for idx in top_indices:
        sim_word = i_to_w[idx]
        if sim_word != word: # Skip the word itself
            print(f"  → {sim_word}: {similarities[idx].item():.4f}")

#Example Usage
find_most_similar("king", model, word_to_idx, idx_to_word, top_n=5)
find_most_similar("apple", model, word_to_idx, idx_to_word, top_n=5)
