import torch
import torch.nn as nn
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

with open('shakespeare.txt') as f:
	file = f.read()

tokens = list(set(file)) + ['<START>', '<END>', '<MASK>']
encoder = {v : i for i, v in enumerate(tokens)}
decoder = {i : v for i, v in enumerate(tokens)}

encoded_dataset = []

for c in file:
	encoded_dataset.append(encoder[c])

encoded_dataset = torch.tensor(encoded_dataset)

#### Constants
CONTEXT_WINDOW = 16
MASK_PERCENTAGE = 0.15
VOCABULARY_SIZE = len(tokens)
MASKS_PER_SEQUENCE = int(CONTEXT_WINDOW * MASK_PERCENTAGE)
EMBEDDING_SIZE = 64
NUM_HEADS      = 4
HEAD_SIZE      = EMBEDDING_SIZE // NUM_HEADS
BATCH_SIZE     = 64
NUM_ENCODER_BLOCKS = 4
DROPOUT_RATE = 0.1
EPOCHS = 1000

#### Preprocessing
num_sequences = encoded_dataset.shape[0] // CONTEXT_WINDOW

X = torch.zeros(size=(num_sequences, CONTEXT_WINDOW), dtype=torch.long)
Y = torch.zeros(size=(num_sequences, MASKS_PER_SEQUENCE), dtype=torch.long)

for i in range(num_sequences):
	X[i][0]    = encoder['<START>']
	X[i][1:-1] = encoded_dataset[i*CONTEXT_WINDOW:(i+1)*CONTEXT_WINDOW-2]
	X[i][-1]   = encoder['<END>']

### BERT

class FeedForward(nn.Module):
	def __init__(self):
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE),
			nn.Dropout(DROPOUT_RATE),
			nn.ReLU(),
			nn.Dropout(DROPOUT_RATE),
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
		self.dropout = nn.Dropout(DROPOUT_RATE)

	def forward(self, data):
		head_outputs = [head(data) for head in self.attention_heads]
		concatenated_outputs = torch.cat(head_outputs, dim=-1)
		projected_outputs = self.linear(concatenated_outputs)
		return self.dropout(projected_outputs)

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
	
	def generate(self, num_iterations=10):
		self.eval()

		# Initialize with START, END, and MASKs in between
		generated_text = torch.full(size=(1, CONTEXT_WINDOW), fill_value=encoder['<MASK>'], dtype=torch.long)
		generated_text[0, 0] = encoder['<START>']
		generated_text[0, -1] = encoder['<END>']

		for _ in range(num_iterations):
			logits = self(generated_text)
			probs = nn.functional.softmax(logits, dim=-1)

			for pos in range(1, CONTEXT_WINDOW - 1):
				# Only update positions that are still MASK
				if generated_text[0, pos] == encoder['<MASK>']:
					sampled_token = torch.multinomial(probs[0, pos], num_samples=1).item()

					# Avoid regenerating MASK
					if sampled_token != encoder['<MASK>']:
						generated_text[0, pos] = sampled_token

		# Decode final output
		decoded = ''.join(decoder[tok.item()] for tok in generated_text[0])
		print(decoded)



def get_batch():
	batch_indices = torch.randint(low=0, high=X.shape[0] - 1, size=(BATCH_SIZE,))
	X_batch = X[batch_indices].clone()
	Y_batch = torch.zeros(size=(BATCH_SIZE, MASKS_PER_SEQUENCE), dtype=torch.long)
	masked_indices = torch.randint(low=1, high=CONTEXT_WINDOW - 1, size=(BATCH_SIZE, MASKS_PER_SEQUENCE), dtype=torch.long)

	for i in range(BATCH_SIZE):
		mask_pos = masked_indices[i]
		Y_batch[i] = X_batch[i][mask_pos]
		X_batch[i][mask_pos] = encoder['<MASK>']

	return X_batch, Y_batch, masked_indices


model = BERT()

optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
model.train()

for i in range(EPOCHS):
	x, y, indices = get_batch()

	logits = model(x)

	batch_indices = torch.arange(x.shape[0]).unsqueeze(1)
	masked_logits = logits[batch_indices, indices]

	masked_logits = masked_logits.reshape(-1, masked_logits.size(-1))  # [B*M, V]
	y_flat = y.reshape(-1)

	loss = loss_fn(masked_logits, y_flat)

	optim.zero_grad()
	loss.backward()
	optim.step()

	if i % 10 == 0: print(f'Loss at epoch {i} = {loss.item():.4f}')

pca = PCA(n_components=2)
embeddings = pca.fit_transform(model.token_embedding.weight.detach().cpu().numpy())

plt.scatter(embeddings[:, 0], embeddings[:, 1])
plt.title("PCA of token embeddings")

for i, (x, y) in enumerate(embeddings[:, :2]):
    plt.text(x+0.1, y+0.1, decoder[i], fontsize=9)

plt.show()

embeddings = pca.fit_transform(model.positional_embedding.weight.detach().cpu().numpy())

plt.scatter(embeddings[:, 0], embeddings[:, 1])
plt.title("PCA of positional embeddings")

for i, (x, y) in enumerate(embeddings[:, :2]):
    plt.text(x+0.1, y+0.1, i, fontsize=9)

plt.show()