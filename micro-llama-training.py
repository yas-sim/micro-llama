import torch
import time
import pandas as pd
import urllib.request
from tqdm import tqdm

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"   # The URL of the raw text file on GitHub
file_name = "tinyshakespeare.txt"                                                   # The file name for local storage
urllib.request.urlretrieve(url, file_name)                                          # Execute the download
lines = open("tinyshakespeare.txt", 'r').read()                                     # Read the content of the dataset
vocab = sorted(list(set(lines)))                                                    # Create a sorted list of unique characters in the dataset
print('Printing the first 10 characters of the vocab list:', vocab[:10])            # Display the first 10 characters in the vocabulary list
print('Total number of characters in our dataset (Vocabulary Size):', len(vocab))   # Output the total number of characters in our dataset (Vocabulary Size)

# Model configurations for the number of layers
MASTER_CONFIG = {
    'vocab_size'    : len(vocab),
    'context_window': 16,       # Number of characters in each input (x) and target (y) sequence of each batch
    'd_model'       : 128,
    'n_heads'       : 8,
    "log_interval"  : 10,
    'n_layers'      : 4,        # Set the number of layers to 4
    'epochs'        : 10000,
    'batch_size'    : 32,       # Increase batch size to 32
}

itos = {i: ch for i, ch in enumerate(vocab)}    # Mapping integers to characters (itos)
stoi = {ch: i for i, ch in enumerate(vocab)}    # Mapping characters to integers (stoi)

def encode(s):                                  # Encode function: Converts a string to a list of integers using the mapping stoi
    return [stoi[ch] for ch in s]

def decode(l):                                  # Decode function: Converts a list of integers back to a string using the mapping itos
    return ''.join([itos[i] for i in l])

dataset = torch.tensor(encode(lines), dtype=torch.int8)     # Convert the dataset into a torch tensor with specified data type (dtype)
print(dataset.shape)                                        # Display the shape of the resulting tensor

# Function to get batches for training, validation, or testing
def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    # Split the dataset into training, validation, and test sets
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]

    # Determine which split to use
    batch_data = train
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test

    # Pick random starting points within the data
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))

    # Create input sequences (x) and corresponding target sequences (y)
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()

    return x, y

@torch.no_grad()  # Don't compute gradients for this function
def evaluate_loss(model, config=MASTER_CONFIG):
    out = {}                                    # Placeholder for the evaluation results
    model.eval()                                # Set the model to evaluation mode
    for split in ["train", "val"]:              # Iterate through training and validation splits
        losses = []                             # Placeholder for individual losses
        for _ in range(10):                     # Generate 10 batches for evaluation
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])                # Get input sequences (xb) and target sequences (yb)
            _, loss = model(xb, yb)             # Perform model inference and calculate the loss
            losses.append(loss.item())          # Append the loss to the list
        out[split] = np.mean(losses)            # Calculate the mean loss for the split and store it in the output dictionary
    model.train()                               # Set the model back to training mode
    return out

# Function to perform training
def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    losses = []                                         # Placeholder for storing losses
    start_time = time.time()                            # Start tracking time
    for epoch in tqdm(range(config['epochs'])):         # Iterate through epochs
        optimizer.zero_grad()                           # Zero out gradients
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])  # Obtain batches for training
        logits, loss = model(xs, targets=ys)            # Forward pass through the model to calculate logits and loss
        loss.backward()                                 # Backward pass and optimization step
        optimizer.step()

        if scheduler:                                   # If a learning rate scheduler is provided, adjust the learning rate
            scheduler.step()

        if epoch % config['log_interval'] == 0:         # Log progress every specified interval
            batch_time = time.time() - start_time       # Calculate batch time
            x = evaluate_loss(model)                    # Evaluate loss on validation set
            losses += [x]                               # Store the validation loss
            
            if print_logs:                              # Print progress logs if specified
                print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")
                
            start_time = time.time()                    # Reset the timer

            if scheduler:                               # Print learning rate if a scheduler is provided
                print("lr: ", scheduler.get_lr())

    print("Validation loss: ", losses[-1]['val'])       # Print the final validation loss
    return pd.DataFrame(losses).plot()                  # Plot the training and validation loss curves

from model_llama import *
#"""
llama = Llama(MASTER_CONFIG)                                    # Create an instance of RopeModel (RMSNorm, RoPE, Multi-Head, SwiGLU, N_layers)

optimizer = torch.optim.Adam(llama.parameters())                # Define the Adam optimizer for model parameters
train(llama, optimizer)                                         # Train the model
#train(llama, optimizer, scheduler=None, config=MASTER_CONFIG)   # Train the LLaMA model for the specified number of epochs

xs, ys = get_batches(dataset, 'test', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window']) # Get batches from the test set
logits, loss = llama(xs, ys)                                    # Pass the test data through the LLaMA model
print(loss)                                                     # Print the loss on the test set
torch.save(llama, 'llama_model.pth')                            # Save the entire model
torch.save(llama.state_dict(), 'llama_model_params.pth')        # If you want to save only the model parameters

"""
MASTER_CONFIG.update({"epochs": 10000})                         # Update configuration
llama_with_cosine = Llama(MASTER_CONFIG)                        # Create Llama model with Cosine Annealing learning schedule
llama_optimizer = torch.optim.Adam(llama_with_cosine.parameters(), betas=(.9, .95), weight_decay=.1, eps=1e-9, lr=1e-3) # Define Adam optimizer with specific hyperparameters
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(llama_optimizer, 300, eta_min=1e-5)  # Define Cosine Annealing learning rate scheduler
train(llama_with_cosine, llama_optimizer, scheduler=scheduler)  # Train the Llama model with the specified optimizer and scheduler

xs, ys = get_batches(dataset, 'test', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window']) # Get batches from the test set
logits, loss = llama_with_cosine(xs, ys)                        # Pass the test data through the LLaMA model
print(loss)                                                     # Print the loss on the test set

torch.save(llama_with_cosine, 'llama_model_cosine.pth')         # Save the entire model
torch.save(llama_with_cosine.state_dict(), 'llama_model_params_cosine.pth')                 # If you want to save only the model parameters
"""