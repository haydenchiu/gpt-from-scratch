from datasets import load_dataset  # huggingface data loading function
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """Tokenize each text sample and return input IDs and attention mask"""
        # return self.data[index]
        # Tokenize each text sample and return input IDs and attention mask
        tokenized = self.tokenizer(
            self.data[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze()
        }
    
def get_wikitext_data(tokenizer_name="gpt2", max_length=512):
    """Load raw text data from wikitext dataset"""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = dataset["train"]["text"]
    valid_texts = dataset["validation"]["text"]
    test_texts = dataset["test"]["text"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure compatibility for padding

    # Wrap text data in the TextDataset with tokenization
    train_data = TextDataset(train_texts, tokenizer, max_length=max_length)
    valid_data = TextDataset(valid_texts, tokenizer, max_length=max_length)
    test_data = TextDataset(test_texts, tokenizer, max_length=max_length)

    return train_data, valid_data, test_data

if __name__ == "__main__":
    # Testing get_wikitext_data
    train_data, valid_data, test_data = get_wikitext_data()
    
    # Sample data loader
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    sample_batch = next(iter(train_loader))

    print("Sample batch keys:", sample_batch.keys())
    print("Sample input_ids batch shape:", sample_batch["input_ids"].shape)
    print("Sample attention_mask batch shape:", sample_batch["attention_mask"].shape)