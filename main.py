import torch

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
CONTEXT_WINDOW = 64
MASK_PERCENTAGE = 0.15
VOCABULARY_SIZE = len(tokens)
MASKS_PER_SEQUENCE = int(CONTEXT_WINDOW * MASK_PERCENTAGE)

#### Preprocessing
num_sequences = encoded_dataset.shape[0] // CONTEXT_WINDOW

X = torch.zeros(size=(num_sequences, CONTEXT_WINDOW), dtype=torch.long)
Y = torch.zeros(size=(num_sequences, MASKS_PER_SEQUENCE, VOCABULARY_SIZE), dtype=torch.long)

for i in range(num_sequences):
	X[i][0]    = encoder['<START>']
	X[i][1:-1] = encoded_dataset[i*CONTEXT_WINDOW:(i+1)*CONTEXT_WINDOW-2]
	X[i][-1]   = encoder['<END>']


masked_indices = torch.randint(low=1, high=CONTEXT_WINDOW-1, size=(num_sequences, MASKS_PER_SEQUENCE), dtype=torch.long)

for i in range(masked_indices.shape[0]):
	indices = masked_indices[i]
	original_tokens = X[i][indices]
	for j, k in enumerate(original_tokens):
		Y[i, j, int(k)] = 1
	X[i][indices] = encoder['<MASK>']
