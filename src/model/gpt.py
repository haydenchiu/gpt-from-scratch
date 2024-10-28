import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print ("MPS device found.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print ("cuda device found.")
else:
    device = torch.device("cpu")
    print ("cpu device found.")

class SingleHeadedAttention(nn.Module):
    def __init__(self, model_dim: int, head_size:int):
        # model_dim equals embedding_dim
        super().__init__()
        self.query_layer = nn.Linear(model_dim, head_size, bias=False)
        self.key_layer = nn.Linear(model_dim, head_size, bias=False)
        self.value_layer = nn.Linear(model_dim, head_size, bias=False)

    def forward(self, embedded):
        embedded = embedded.to(device)
        q = self.query_layer(embedded).to(device) # B x T x A
        k = self.key_layer(embedded).to(device) # B x T x A
        v = self.value_layer(embedded).to(device) # B x T x A

        # Scores: (B x T x A) @ (B x A x T) -> B x T x T
        scores = q @ torch.transpose(k, 1, 2)

        context_length, attention_dim = k.shape[1], k.shape[2]
        scores = scores / (attention_dim ** 0.5)

        # Masking
        lower_tri = torch.tril(torch.ones((context_length, context_length), device=device))
        mask = (lower_tri == 0).to(device)
        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax
        scores = F.softmax(scores, dim=2)

        # Attention output
        attention_output = scores @ v
        return attention_output
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = model_dim // num_heads
        self.attention_heads = nn.ModuleList()
        for i in range(num_heads):
            self.attention_heads.append(SingleHeadedAttention(model_dim, self.head_size))
        self.compute = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, embedded):
        embedded = embedded.to(device)
        head_outputs = []
        for head in self.attention_heads:
            head_outputs.append(head(embedded))
        concat = torch.cat(head_outputs, dim=-1)
        return self.dropout(self.compute(concat))
    
class VanillaNeuralNetwork(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.first_linear_layer = nn.Linear(model_dim, model_dim * 4)
        self.relu_layer = nn.ReLU()
        self.second_linear_layer = nn.Linear(model_dim * 4, model_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.to(device)
        return self.dropout(self.second_linear_layer(self.relu_layer(self.first_linear_layer(x))))
    
class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.norm_layer_1 = nn.LayerNorm(model_dim)
        self.norm_layer_2 = nn.LayerNorm(model_dim)
        self.mha = MultiHeadedAttention(model_dim, num_heads)
        self.feed_forward = VanillaNeuralNetwork(model_dim)

    def forward(self, embedded):
        embedded = embedded.to(device)
        mha_output = self.mha(self.norm_layer_1(embedded))
        embedded = embedded + mha_output
        feed_forward_output = self.feed_forward(self.norm_layer_2(embedded))
        embedded = embedded + feed_forward_output
        return embedded.to(device)
    
class GPT(nn.Module):
    def __init__(self, model_dim, num_heads, vocab_size, context_length, num_blocks):
        super().__init__()
        self.word_emdedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(context_length, model_dim)
        self.transformer_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.transformer_blocks.append(TransformerBlock(model_dim, num_heads))
        self.norm_layer_3 = nn.LayerNorm(model_dim)
        self.last_linear_layer = nn.Linear(model_dim, vocab_size)

    def forward(self, context):
        context = context.to(device)
        
        # Embedding
        embedded = self.word_emdedding(context)
        context_length = context.shape[1]
        positions = torch.arange(context_length).unsqueeze(0).to(device)
        embedded = embedded + self.position_embedding(positions)
        
        # Transformer blocks
        output = self.transformer_blocks(embedded)
        
        # Output layer
        output = self.norm_layer_3(output)
        output = self.last_linear_layer(output)
        
        return output
    
def generate(model, new_chars: int, context, context_length: int, int_to_char: dict) -> str:
    res = []
    for i in range(new_chars):
        if context.shape[1] > context_length:
            context = context[:, -context_length:]
        prediction = model(context).to(device) # B, T, Vocab_Size
        print(f"Prediction shape: {prediction.shape}")
        last_time_step = prediction[:, -1, :] # B, Vocab_Size
        probabilities = F.softmax(last_time_step, dim = -1)
        next_char = torch.multinomial(probabilities, 1)
        context = torch.cat((context, next_char), dim = -1)
        res.append(int_to_char[next_char.item()])
        print(f"Generated character: {int_to_char[next_char.item()]}")  # Debug new character
    return ''.join(res)
    
if __name__ == "__main__":
    vocab_size = 104
    context_length = 128
    model_dim = 252
    num_blocks = 6
    num_heads = 6

    #model_dim, num_heads, vocab_size, context_length, num_blocks
    model = GPT(model_dim, num_heads, vocab_size, context_length, num_blocks).to(device)

    model.eval()
    new_chars = 5000
    context = torch.zeros(1, 3, dtype = torch.int64).to(device)

    int_to_char = {0: '\n', 1: ' ', 2: '!', 3: '"', 4: '$', 5: '%', 6: '&', 7: "'", 8: '(', 9: ')', 10: '*', 11: '+', 12: ',', 13: '-', 14: '.', 15: '/', 16: '0', 17: '1', 18: '2', 19: '3', 20: '4', 21: '5', 22: '6', 23: '7', 24: '8', 25: '9', 26: ':', 27: ';', 28: '?', 29: 'A', 30: 'B', 31: 'C', 32: 'D', 33: 'E', 34: 'F', 35: 'G', 36: 'H', 37: 'I', 38: 'J', 39: 'K', 40: 'L', 41: 'M', 42: 'N', 43: 'O', 44: 'P', 45: 'Q', 46: 'R', 47: 'S', 48: 'T', 49: 'U', 50: 'V', 51: 'W', 52: 'X', 53: 'Y', 54: 'Z', 55: '[', 56: ']', 57: '_', 58: 'a', 59: 'b', 60: 'c', 61: 'd', 62: 'e', 63: 'f', 64: 'g', 65: 'h', 66: 'i', 67: 'j', 68: 'k', 69: 'l', 70: 'm', 71: 'n', 72: 'o', 73: 'p', 74: 'q', 75: 'r', 76: 's', 77: 't', 78: 'u', 79: 'v', 80: 'w', 81: 'x', 82: 'y', 83: 'z', 84: '{', 85: '|', 86: '}', 87: 'à', 88: 'á', 89: 'è', 90: 'é', 91: 'ë', 92: 'ñ', 93: 'ó', 94: 'ú', 95: '\u2005', 96: '–', 97: '—', 98: '‘', 99: '’', 100: '“', 101: '”', 102: '…', 103: '\u205f'}

    print(generate(model, new_chars, context, context_length, int_to_char))