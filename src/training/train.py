from src.model.gpt import GPT
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from src.data.load_data import *
import os

# Setup device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Define model
vocab_size = 104
context_length = 128
model_dim = 252
num_blocks = 6
num_heads = 6

# model_dim, num_heads, vocab_size, context_length, num_blocks
model = GPT(model_dim, num_heads, vocab_size, context_length, num_blocks).to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

def train(model, data_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        # batch = batch.to(device)
        batch = batch["input_ids"].to(device)
        
        # Shift input for language modeling
        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]

        # Forward pass
        outputs = model(input_ids)
        
        # Calculate loss
        loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1))
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # batch = batch.to(device)
            batch = batch["input_ids"].to(device)
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            # Forward pass
            outputs = model(input_ids)
            
            # Calculate loss
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

if __name__ == "__main__":

    # Load your datasets
    train_data, valid_data, test_data = get_wikitext_data()

    # Set up dataloaders
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Training loop parameters
    epochs = 3
    best_valid_loss = float('inf')
    save_path = 'gpt_model.pth'

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training step
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validation step
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        print(f"Validation Loss: {valid_loss:.4f}")
        
        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            # best_model = model
            print(f"Best model saved with validation loss: {valid_loss:.4f}")

    # Load the best model
    model.load_state_dict(torch.load(save_path))

    # Evaluate on test data
    test_loss = evaluate(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}")

