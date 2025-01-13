import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.config.config import TRAINING_CONFIG, MODEL_CONFIG

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
        q = self.query_layer(embedded)  # B x T x A
        k = self.key_layer(embedded)  # B x T x A
        v = self.value_layer(embedded)  # B x T x A

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
        embedded = embedded
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
        return self.dropout(self.second_linear_layer(self.relu_layer(self.first_linear_layer(x))))
    

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.norm_layer_1 = nn.LayerNorm(model_dim)
        self.norm_layer_2 = nn.LayerNorm(model_dim)
        self.mha = MultiHeadedAttention(model_dim, num_heads)
        self.feed_forward = VanillaNeuralNetwork(model_dim)

    def forward(self, embedded):
        mha_output = self.mha(self.norm_layer_1(embedded))
        embedded = embedded + mha_output
        feed_forward_output = self.feed_forward(self.norm_layer_2(embedded))
        embedded = embedded + feed_forward_output
        return embedded
    

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
    

# @torch.no_grad()
# def generate(model, new_chars: int, context, context_length: int, int_to_char: dict) -> str:
#     model.eval()  # Set to eval mode for generation
#     res = []
#     for i in range(new_chars):
#         if context.shape[1] > context_length:
#             context = context[:, -context_length:]
#         prediction = model(context).to(device)  # B, T, Vocab_Size
#         # print(f"Prediction shape: {prediction.shape}")
#         last_time_step = prediction[:, -1, :]  # B, Vocab_Size
#         probabilities = F.softmax(last_time_step, dim=-1)
#         next_char = torch.multinomial(probabilities, 1)
#         context = torch.cat((context, next_char), dim=1)
#         res.append(int_to_char[next_char.item()])
#         # print(f"Generated character: {int_to_char[next_char.item()]}")  # Debug new character
#     return ''.join(res)


# def generate(model, new_chars: int, context, context_length: int, tokenizer, temperature=MODEL_CONFIG["temperature"], top_k=MODEL_CONFIG["top_k"]):
#     generated = []
#     model.eval()
    
#     with torch.no_grad():
#         for _ in range(new_chars):
#             if context.size(1) > context_length:
#                 context = context[:, -context_length:]  # Ensure context doesn't exceed max length

#             # Get logits for the last token in the sequence (last time step)
#             logits = model(context)[:, -1, :]  # Shape: [batch_size, vocab_size]
            
#             # Apply temperature scaling
#             logits = logits / temperature
            
#             # Apply top-k sampling to restrict the range of possible next tokens
#             top_k_logits, top_k_indices = torch.topk(logits, top_k)
#             top_k_probs = F.softmax(top_k_logits, dim=-1)
            
#             # Sample from top-k tokens using probabilities
#             sampled_idx = torch.multinomial(top_k_probs, 1).item()  # Get the index of the chosen token in the top-k list
#             # next_token = top_k_indices[0, sampled_idx]  # Extract the sampled token index from top_k_indices
#             next_token = top_k_indices[0, sampled_idx].item()  # Extract the sampled token index from top_k_indices


#             # Append next token to the context for the next iteration
#             # context = torch.cat([context, next_token.unsqueeze(0).unsqueeze(0)], dim=1)  # Shape: [1, seq_len + 1]
#             context = torch.cat([context, torch.tensor([[next_token]], device=context.device)], dim=1)

#             # Add the generated token to the list
#             # generated.append(next_token.item())
#             generated.append(next_token)
    
#     # Convert the generated token IDs to text using the tokenizer
#     return tokenizer.decode(generated, skip_special_tokens=True)


def generate(model, new_chars: int, context, context_length: int, tokenizer, temperature=MODEL_CONFIG["temperature"], top_k=MODEL_CONFIG["top_k"], top_p=MODEL_CONFIG["top_p"]):
    generated = []
    model.eval()
    
    with torch.no_grad():
        for _ in range(new_chars):
            if context.size(1) > context_length:
                context = context[:, -context_length:]  # Ensure context doesn't exceed max length

            # Get logits for the last token in the sequence (last time step)
            logits = model(context)[:, -1, :]  # Shape: [batch_size, vocab_size]

            # Apply temperature scaling
            logits = logits / temperature

            # Debug print: Check logits before any masking
            # print("Logits before masking:\n", logits)

            # Check for NaN or Inf values in logits before proceeding
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Logits contain NaN or Inf values before masking.")
                return None  # Early exit if there's an issue

            # Apply top-k and top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Create a mask for tokens with cumulative probability above the threshold (top-p)
            sorted_indices_to_remove = cumulative_probs > top_p

            # Shift the indices to remove by 1 to maintain only those that exceed top_p threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Apply top-k filtering
            if top_k > 0:
                top_k_cutoff = min(top_k, sorted_logits.size(-1))  # Ensure we don't exceed vocabulary size
                sorted_indices_to_remove[..., top_k_cutoff:] = True

            # Initialize the mask with -Inf for the filtered indices
            mask = torch.full_like(logits, fill_value=-float('Inf'))
            mask[sorted_indices[~sorted_indices_to_remove]] = logits[sorted_indices[~sorted_indices_to_remove]]

            # Debug print: Check logits after masking
            # print("Logits after masking (before clamping):\n", mask)

            # Update logits with the mask
            logits = mask

            # **Clamping**: Limit the logits range to avoid extreme values
            logits = torch.clamp(logits, min=-10, max=10)

            # Debug print: Check logits after clamping
            # print("Logits after clamping:\n", logits)

            # Check for NaN or Inf values in logits after masking and clamping
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Logits contain NaN or Inf values after masking and clamping.")
                return None  # Early exit if there's an issue

            # Compute the probabilities from the logits
            probabilities = F.softmax(logits, dim=-1)

            # Check for NaN or Inf values in probabilities before sampling
            if torch.isnan(probabilities).any() or torch.isinf(probabilities).any() or (probabilities < 0).any():
                print("Probabilities contain NaN, Inf, or negative values.")
                return None  # Early exit if there's an issue

            # Sample from the filtered distribution
            next_token = torch.multinomial(probabilities, 1)  # Shape: [batch_size=1, 1]

            # Ensure next_token has shape [1, 1]
            next_token = next_token.unsqueeze(0) if next_token.dim() == 1 else next_token

            # Reshape next_token to be 2D if it's 3D
            if next_token.dim() == 3:
                next_token = next_token.squeeze(dim=-1)  # Shape should be [1, 1]

            # Append next token to the context for the next iteration
            context = torch.cat([context, next_token], dim=1)  # Shape: [1, seq_len + 1]
            
            # Add the generated token to the list
            generated.append(next_token.item())
    
    # Convert the generated token IDs to text using the tokenizer
    return tokenizer.decode(generated)



if __name__ == "__main__":
    vocab_size = 50257
    context_length = 512
    model_dim = 252
    num_blocks = 6
    num_heads = 6
    tokenizer_name = 'gpt2'

    # model_dim, num_heads, vocab_size, context_length, num_blocks
    model = GPT(model_dim, num_heads, vocab_size, context_length, num_blocks).to(device)

    new_chars = 500
    context = torch.zeros(1, 3, dtype=torch.long, device=device)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure compatibility for padding

    # int_to_char = {0: '\n', 1: ' ', 2: '!', 3: '"', 4: '$', 5: '%', 6: '&', 7: "'", 8: '(', 9: ')', 10: '*', 11: '+', 12: ',', 13: '-', 14: '.', 15: '/', 16: '0', 17: '1', 18: '2', 19: '3', 20: '4', 21: '5', 22: '6', 23: '7', 24: '8', 25: '9', 26: ':', 27: ';', 28: '?', 29: 'A', 30: 'B', 31: 'C', 32: 'D', 33: 'E', 34: 'F', 35: 'G', 36: 'H', 37: 'I', 38: 'J', 39: 'K', 40: 'L', 41: 'M', 42: 'N', 43: 'O', 44: 'P', 45: 'Q', 46: 'R', 47: 'S', 48: 'T', 49: 'U', 50: 'V', 51: 'W', 52: 'X', 53: 'Y', 54: 'Z', 55: '[', 56: ']', 57: '_', 58: 'a', 59: 'b', 60: 'c', 61: 'd', 62: 'e', 63: 'f', 64: 'g', 65: 'h', 66: 'i', 67: 'j', 68: 'k', 69: 'l', 70: 'm', 71: 'n', 72: 'o', 73: 'p', 74: 'q', 75: 'r', 76: 's', 77: 't', 78: 'u', 79: 'v', 80: 'w', 81: 'x', 82: 'y', 83: 'z', 84: '{', 85: '|', 86: '}', 87: 'à', 88: 'á', 89: 'è', 90: 'é', 91: 'ë', 92: 'ñ', 93: 'ó', 94: 'ú', 95: '\u2005', 96: '–', 97: '—', 98: '‘', 99: '’', 100: '“', 101: '”', 102: '…', 103: '\u205f'}
    # int_to_char = {i: chr(65+i) for i in range(vocab_size)}


    print(generate(model, new_chars, context, context_length, tokenizer))