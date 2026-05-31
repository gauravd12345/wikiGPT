import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"eos_token": "<eos>", "additional_special_tokens": ["<sep>"]})

""" Hyper parameter """

d_model = 512
h = 8
d_k = 64
d_v = 64
N = 4

lr = 2.5e-4
epochs = 20
batch_size = 64
vocab_size = len(tokenizer)
block_size = 128

class DecoderTransformer(nn.Module):
  def __init__(self):
    super().__init__()

    self.embed = nn.Embedding(vocab_size, d_model)
    self.pos = nn.Embedding(block_size, d_model) # positional embedding

    self.W_q = nn.ModuleList([nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(h)]) for _ in range(N)]) # q, k, v projections
    self.W_k = nn.ModuleList([nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(h)]) for _ in range(N)])
    self.W_v = nn.ModuleList([nn.ModuleList([nn.Linear(d_model, d_v) for _ in range(h)]) for _ in range(N)])

    self.W_o = nn.ModuleList([nn.Linear(h * d_v, d_model) for _ in range(N)]) # final projection

    self.ffn1 = nn.ModuleList([nn.Linear(d_model, 4 * d_model) for _ in range(N)]) # ffn layer
    self.ffn2 = nn.ModuleList([nn.Linear(4 * d_model, d_model) for _ in range(N)])

    self.fc = nn.Linear(d_model, vocab_size) # final layer

    self.ln1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(N)])
    self.ln2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(N)])

    self.dropout = nn.Dropout(0.1)

  def multi_head_attention(self, x, layer_idx):
    W_tot = []
    mask = torch.triu(torch.full((x.size(1), x.size(1)), float('-inf')), diagonal=1).to(x.device)
    for Q, K, V in zip(self.W_q[layer_idx], self.W_k[layer_idx], self.W_v[layer_idx]):
      Q_i = Q(x)
      K_i = K(x)
      V_i = V(x)

      alignment = torch.matmul(Q_i, K_i.transpose(-2, -1))         # query-key alignment
      alignment = alignment + mask                                 # adding mask

      wei = self.dropout(torch.softmax(alignment / (d_k ** 0.5), dim=-1))         # alignment weights
      wei_value = torch.matmul(wei, V_i)                           # weighted values

      W_tot.append(wei_value)

    out = self.W_o[layer_idx](torch.cat(W_tot, dim=2))
    return out

  def forward(self, x): # (batch_size, block_size)
    p = torch.arange(x.size(1)).to(x.device)
    x = self.dropout(self.embed(x) + self.pos(p))     # word & positional embedding

    for i in range(N):
      out = self.multi_head_attention(x, i)  # multi head attention
      out = self.ln1[i](out + x)             # layernorm + residual connection

      fn = self.ffn1[i](out)                 # ffn
      fn = torch.relu(fn)
      fn = self.dropout(self.ffn2[i](fn))

      out = self.ln2[i](fn + out)
      x = out

    out = self.fc(out)
    return out


# load model
model = DecoderTransformer().to(device)
model.load_state_dict(torch.load("/Users/gauravd/Downloads/model.pth", map_location=device))
model.eval()

sep_token_id = tokenizer.convert_tokens_to_ids("<sep>")

def sample(logits, temperature=0.8):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

def chat(prompt, seq_len=30):
    test = prompt + " <sep>"
    with torch.no_grad():
        input_tokens = tokenizer.encode(test)
        tokens = input_tokens.copy()
        for i, _ in enumerate(range(seq_len)):
            x = tokens[-block_size:]
            xb = torch.tensor(x).unsqueeze(0).to(device)
            out = model(xb)
            logits = out[0, -1]
            if i == 0:
                logits[sep_token_id] = float('-inf')
            logits[tokenizer.eos_token_id] = float('-inf')
            next_token = sample(logits)
            tokens.append(next_token)
            if next_token == sep_token_id:
                break
    return tokenizer.decode(tokens[len(input_tokens):]).replace("<sep>", "").strip()

# chatbot loop
print("Chatbot ready! Type 'quit' to exit.")

history = []
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    history.append(user_input)
    prompt = " <sep> ".join(history)
    response = chat(prompt)
    history.append(response)
    print(f"Bot: {response}")

