import torch
import torch.nn as nn
from math import sqrt

with open('shakespeare.txt') as f:
	file = f.read()

tokens = list(set(file)) + ['<START>', '<END>', '<MASK>']
print(tokens)
encoder = {v : i for i, v in enumerate(tokens)}
decoder = {i : v for i, v in enumerate(tokens)}

encoded_dataset = []

for c in file:
	encoded_dataset.append(encoder[c])

encoded_dataset = torch.tensor(encoded_dataset)

#### Constants
CONTEXT_WINDOW = 64
MASK_PERCENTAGE = 0.15
VOCABULARY_SIZE = len(tokens)
MASKS_PER_SEQUENCE = int(CONTEXT_WINDOW * MASK_PERCENTAGE)
EMBEDDING_SIZE = 128
NUM_HEADS      = 8
HEAD_SIZE      = EMBEDDING_SIZE // NUM_HEADS
BATCH_SIZE     = 8
NUM_ENCODER_BLOCKS = 4

#### Preprocessing
num_sequences = encoded_dataset.shape[0] // CONTEXT_WINDOW

X = torch.zeros(size=(num_sequences, CONTEXT_WINDOW), dtype=torch.long)
Y = torch.zeros(size=(num_sequences, MASKS_PER_SEQUENCE), dtype=torch.long)

for i in range(num_sequences):
	X[i][0]    = encoder['<START>']
	X[i][1:-1] = encoded_dataset[i*CONTEXT_WINDOW:(i+1)*CONTEXT_WINDOW-2]
	X[i][-1]   = encoder['<END>']


masked_indices = torch.randint(low=1, high=CONTEXT_WINDOW-1, size=(num_sequences, MASKS_PER_SEQUENCE), dtype=torch.long)

for i in range(masked_indices.shape[0]):
	indices = masked_indices[i]
	original_tokens = X[i][indices]
	for j, k in enumerate(original_tokens):
		Y[i, j] = int(k)
	X[i][indices] = encoder['<MASK>']

### BERT

class FeedForward(nn.Module):
	def __init__(self):
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE),
			nn.ReLU(),
			nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
		)

	def forward(self, data):
		return self.network(data)
	
class AttentionHead(nn.Module):
	def __init__(self):
		super().__init__()
		self.key = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
		self.query = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
		self.value = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, data):		
		key = self.key(data)
		query = self.query(data)
		value = self.value(data)

		mat_mul = query @ key.transpose(-2, -1) # Dot Product
		scaled_mat_mul = self.softmax(mat_mul * (1 / sqrt(HEAD_SIZE)))
		return scaled_mat_mul @ value


class MultiHeadedAttention(nn.Module):
	def __init__(self):
		super().__init__()
		self.attention_heads = nn.ModuleList([AttentionHead() for i in range(NUM_HEADS)])
		self.linear = nn.Linear(HEAD_SIZE * NUM_HEADS, EMBEDDING_SIZE)

	def forward(self, data):
		head_outputs = [head(data) for head in self.attention_heads]
		concatenated_outputs = torch.cat(head_outputs, dim=-1)
		return self.linear(concatenated_outputs)

class EncoderBlock(nn.Module):
	def __init__(self):
		super().__init__()
		self.attention = MultiHeadedAttention()
		self.layer_norm_1 = nn.LayerNorm(EMBEDDING_SIZE)
		self.feed_forward = FeedForward()
		self.layer_norm_2 = nn.LayerNorm(EMBEDDING_SIZE)

	def forward(self, data):
		attention_outputs = self.attention(data)
		normalised_data_1 = self.layer_norm_1(data + attention_outputs)
		normalised_data_2 = self.layer_norm_2(self.feed_forward(normalised_data_1) + normalised_data_1)

		return normalised_data_2



class BERT(nn.Module):
	def __init__(self):
		super().__init__()
		self.token_embedding     = torch.nn.Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE)
		self.positional_embedding = torch.nn.Embedding(CONTEXT_WINDOW, EMBEDDING_SIZE)
		self.encoder_blocks = nn.ModuleList([EncoderBlock() for i in range(NUM_ENCODER_BLOCKS)])
		self.linear = nn.Linear(EMBEDDING_SIZE, VOCABULARY_SIZE)


	def forward(self, data):
		B, T = data.shape

		token_embeddings = self.token_embedding(data)
		positional_embeddings = self.positional_embedding(torch.arange(T, device=data.device)).unsqueeze(0).expand(B, T, -1)
		embedding = token_embeddings + positional_embeddings

		for block in self.encoder_blocks:
			embedding = block(embedding)

		logits = self.linear(embedding)

		return logits


bert = BERT()