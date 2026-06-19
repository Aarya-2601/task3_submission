import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import math

txt="the quick brown fox jumps over the lazy dog"
tokens=txt.lower().split()
counts=Counter(tokens)
vocab=sorted(counts,key=counts.get,reverse=True)
w2i={w:i for i,w in enumerate(vocab)}
i2w={i:w for i,w in enumerate(vocab)}
v_size=len(vocab)

win=2
data=[]
for i,w in enumerate(tokens):
	w_idx=w2i[w]
	start=max(0,i-win)
	end=min(len(tokens),i+win+1)
	for j in range(start,end):
		if i!=j:
			data.append((w_idx,w2i[tokens[j]]))

class Word2Vec(nn.Module):
	def __init__(self,v_size,emb_dim):
		super().__init__()
		self.in_emb=nn.Embedding(v_size,emb_dim)
		self.out_emb=nn.Linear(emb_dim,v_size,bias=False)
	
	def forward(self,tgt):
		v=self.in_emb(tgt)
		scores=self.out_emb(v)
		return scores

emb_dim=10
model=Word2Vec(v_size,emb_dim)
criterion=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(),lr=0.01)

for epoch in range(100):
	tot_loss=0
	for tgt,ctx in data:
		x=torch.tensor([tgt])
		y=torch.tensor([ctx])
		opt.zero_grad()
		out=model(x)
		loss=criterion(out,y)
		loss.backward()
		opt.step()
		tot_loss+=loss.item()
	if(epoch+1)%20==0:
		print(f"Epoch:{epoch+1}, Loss:{tot_loss/len(data):.4f}")

print("\n--- Embedding Results ---")
emb_weights=model.in_emb.weight.detach()
for w in ["fox","dog","the"]:
	idx=w2i[w]
	vec=emb_weights[idx]
	print(f"Word:{w:<5}, Vector length:{torch.norm(vec):.4f}")