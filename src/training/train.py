from src.model.gpt import GPT
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from src.data.load_data import *
from src.config.config import MODEL_CONFIG, TRAINING_CONFIG
import os

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
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()

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
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()

if __name__ == "__main__":
    # Define model
    device  = TRAINING_CONFIG["device"]
    vocab_size = MODEL_CONFIG["vocab_size"]
    context_length = MODEL_CONFIG["context_length"]
    model_dim = MODEL_CONFIG["model_dim"]
    num_blocks = MODEL_CONFIG["num_blocks"]
    num_heads = MODEL_CONFIG["num_heads"]
    learning_rate = TRAINING_CONFIG["learning_rate"]

    # model_dim, num_heads, vocab_size, context_length, num_blocks
    model = GPT(model_dim, num_heads, vocab_size, context_length, num_blocks).to(device)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Load your datasets
    # train_data, valid_data, test_data = get_wikitext_data()
    train_data, valid_data, test_data = get_openwebtext_data()

    # Set up dataloaders
    batch_size = TRAINING_CONFIG["batch_size"]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Training loop parameters
    epochs = TRAINING_CONFIG["num_epochs"]
    best_valid_loss = float('inf')
    save_path = TRAINING_CONFIG["save_path"]

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training step
        train_loss, train_ppl = train(model, train_loader, optimizer, loss_fn, device)
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training PPL: {train_ppl:.4f}")
        
        # Validation step
        valid_loss, valid_ppl = evaluate(model, valid_loader, loss_fn, device)
        print(f"Validation Loss: {valid_loss:.4f}")
        print(f"Validation PPL: {valid_ppl:.4f}")
        
        # Save the best model
        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     best_valid_ppl = valid_ppl
        #     torch.save(model.state_dict(), save_path)
        #     # best_model = model
        #     print(f"Best model saved with validation loss: {valid_loss:.4f}")
        #     print(f"Best model saved with validation PPL: {valid_ppl:.4f}")

    # Load the best model
    # model.load_state_dict(torch.load(save_path))

    # Save model
    torch.save(model.state_dict(), save_path)

    # Evaluate on test data
    test_loss, test_ppl = evaluate(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test PPL: {test_ppl:.4f}")

