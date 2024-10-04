from datasets import load_dataset # huggingface data loading function
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
def get_wikitext_data():
    """Load raw text data from wikitext dataset"""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train = dataset["train"]["text"]
    valid = dataset["validation"]["text"]
    test = dataset["test"]["text"]

    return TextDataset(train), TextDataset(valid), TextDataset(test)

if __name__ == "__main__":
    # for testing get_wikitext_data()
    train_data, valid_data, test_data = get_wikitext_data()
    print("Training data sample: ", train_data[1])
    print("Training data size: ", len(train_data))