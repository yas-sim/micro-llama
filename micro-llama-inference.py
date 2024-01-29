import torch                                        # PyTorch for implementing LLM (No GPU)
from torch.nn import functional as F

file_name = "tinyshakespeare.txt"                   # The file name for local storage
lines = open(file_name, 'r').read()                 # Read the content of the dataset
vocab = sorted(list(set(lines)))                    # Create a sorted list of unique characters in the dataset

# Model configurations for the number of layers
MASTER_CONFIG = {
    'vocab_size': len(vocab),
    'context_window': 16,       # Number of characters in each input (x) and target (y) sequence of each batch
    'd_model': 128,
    'n_heads': 8,
    "log_interval": 10,
    'n_layers': 4,              # Set the number of layers to 4
    'epochs': 10000,
    'batch_size': 32,           # Increase batch size to 32
}

itos = {i: ch for i, ch in enumerate(vocab)}    # Mapping integers to characters (itos)
stoi = {ch: i for i, ch in enumerate(vocab)}    # Mapping characters to integers (stoi)

def encode(s):                                  # Encode function: Converts a string to a list of integers using the mapping stoi
    return [stoi[ch] for ch in s]

def decode(l):                                  # Decode function: Converts a list of integers back to a string using the mapping itos
    return ''.join([itos[i] for i in l])
import sys
# Generate function for text generation using the trained model
def generate(model, config=MASTER_CONFIG, prompt_text=None, max_new_tokens=30, stream=False):
    if prompt_text is None:
        idx = torch.zeros(5, 1).long()
    else:
        input_data = [ [0] + encode(prompt_text) for _ in range(5) ]
        idx = torch.from_numpy(np.array(input_data)).long()
        print(prompt_text, end='', flush=True)
    for itr in range(max_new_tokens):
        logits = model(idx[:, -config['context_window']:])  # Call the model
        last_time_step_logits = logits[:, -1, :]            # all the batches (1), last time step, all the logits
        p = F.softmax(last_time_step_logits, dim=-1)        # softmax to get probabilities
        idx_next = torch.multinomial(p, num_samples=1)      # sample from the distribution to get the next token
        idx = torch.cat([idx, idx_next], dim=-1)            # append to the sequence
        if stream:
            print(decode([idx_next[0].item()]), end='', flush=True)
    if stream:
        print()
    return [decode(x) for x in idx.tolist()]

from model_llama import *
llama = torch.load('llama_model.pth')
#llama = torch.load('llama_model_cosine.pth')

# Generate text using the trained LLM (llama) with a maximum of 500 tokens
stream = True
print('-'*80)
#generated_texts = generate(llama, MASTER_CONFIG, max_new_tokens=300, stream=stream)
generated_texts = generate(llama, MASTER_CONFIG, "ELIZA", max_new_tokens=300, stream=stream)

if not stream:
    for generated_text in generated_texts:
        print('-'*80)
        print(generated_text)
