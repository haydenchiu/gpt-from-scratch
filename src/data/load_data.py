import random
from datasets import load_dataset  # huggingface data loading function
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.config.config import MODEL_CONFIG, TRAINING_CONFIG

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=MODEL_CONFIG['context_length']):
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
    
def get_wikitext_data(tokenizer_name=MODEL_CONFIG["tokenizer_name"], max_length=MODEL_CONFIG['context_length']):
    """Load raw text data from wikitext dataset"""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = dataset["train"]["text"]
    valid_texts = dataset["validation"]["text"]
    test_texts = dataset["test"]["text"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure compatibility for padding

    # Wrap text data in the TextDataset with tokenization
    train_data = TextDataset(train_texts, tokenizer, max_length=max_length)
    valid_data = TextDataset(valid_texts, tokenizer, max_length=max_length)
    test_data = TextDataset(test_texts, tokenizer, max_length=max_length)

    return train_data, valid_data, test_data


def get_openwebtext_data(tokenizer_name="gpt2", max_length=MODEL_CONFIG['context_length'], valid_size=TRAINING_CONFIG["valid_size"], test_size=TRAINING_CONFIG["test_size"], sample_fraction=0.01):
    """Load raw text data from OpenWebText dataset and split into train, validation, and test sets."""
    # Load raw dataset
    dataset = load_dataset("Bingsu/openwebtext_20p")
    train_texts = dataset["train"]["text"]

    # Randomly sample 10% of the data
    sample_size = int(len(train_texts) * sample_fraction)
    train_texts = random.sample(train_texts, sample_size)
    
    # Split train_texts into train, validation, and test sets
    train_texts, test_texts = train_test_split(train_texts, test_size=test_size, random_state=TRAINING_CONFIG["seed"])
    train_texts, valid_texts = train_test_split(train_texts, test_size=valid_size / (1 - test_size), random_state=TRAINING_CONFIG["seed"])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure compatibility for padding

    # Wrap text data in the TextDataset with tokenization
    train_data = TextDataset(train_texts, tokenizer, max_length=max_length)
    valid_data = TextDataset(valid_texts, tokenizer, max_length=max_length)
    test_data = TextDataset(test_texts, tokenizer, max_length=max_length)

    return train_data, valid_data, test_data


def get_char_mappings(tokenizer_name=MODEL_CONFIG["tokenizer_name"]):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure compatibility for padding

    # `char_to_int` maps tokens to their respective ids
    char_to_int = tokenizer.get_vocab()
    # `int_to_char` maps ids back to their tokens
    int_to_char = {v: k for k, v in char_to_int.items()}
    
    return char_to_int, int_to_char


if __name__ == "__main__":
    # Testing get_wikitext_data
    # train_data, valid_data, test_data = get_wikitext_data()
    train_data, valid_data, test_data = get_openwebtext_data()

    # Sample data loader
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    sample_batch = next(iter(train_loader))

    print("Sample batch keys:", sample_batch.keys())
    print("Sample input_ids batch shape:", sample_batch["input_ids"].shape)
    print("Sample attention_mask batch shape:", sample_batch["attention_mask"].shape)