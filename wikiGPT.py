import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# import kagglehub

# # Download latest version
# squad_path = kagglehub.dataset_download("stanfordu/stanford-question-answering-dataset")
# daily_dialog_path = kagglehub.dataset_download("thedevastator/dailydialog-unlock-the-conversation-potential-in")

# print("Path to dataset files:", squad_path)
# print("Path to dataset files:", daily_dialog_path)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"eos_token": "<eos>", "additional_special_tokens": ["<sep>"]})

print(tokenizer.convert_tokens_to_ids("<sep>"))
print(tokenizer.encode("<sep>"))


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


squad_path = "/kaggle/input/datasets/organizations/stanfordu/stanford-question-answering-dataset"
squad_train = squad_path + "/train-v1.1.json"
squad = pd.read_json(squad_train)

daily_dialog_path = "/kaggle/input/datasets/thedevastator/dailydialog-unlock-the-conversation-potential-in"
daily_train = daily_dialog_path + "/train.csv"
daily = pd.read_csv(daily_train)


corpus = []

""" Persona Chat """
persona_chat = pd.read_parquet("hf://datasets/Cynaptics/persona-chat/data/train-00000-of-00001.parquet")
for dialogue in persona_chat['dialogue']:
    dialogue = " <sep> ".join(dialogue)
    
    dialogue = dialogue.replace("Persona A: ", "")
    dialogue = dialogue.replace("Persona B: ", "")
    dialogue += " <eos>"

    corpus.append(dialogue)
    
persona_len = len(corpus)
print(f"Persona Chat Samples: {persona_len}")

""" SQuAD """
max_samples = 20000
count = 0

for article in squad['data']:
    for paragraph in article['paragraphs']:
        context = paragraph['context'].strip()
        for qa in paragraph['qas']:
            question = qa['question'].strip()
            if not qa['answers']:
                continue
            for answer in qa['answers']:
                if count >= max_samples:
                    break
                ans = answer['text'].strip()
                corpus.append(context + " <sep> " + question + " <sep> " + ans + " <eos>")
                count += 1
                
squad_len = len(corpus) - persona_len
print(f"SQaUD Samples: {squad_len}")

""" Daily Dialog """
for dialog in daily['dialog']:
    text = dialog[1:-1]
    utterances = [u.strip().strip("'\"") for u in text.split('\n') if u.strip()]
    utterances = " <sep> ".join(utterances) + " <eos>"

    corpus.append(utterances)

daily_dialog_len = len(corpus) - (squad_len + persona_len)
print(f"Daily Dialog Samples: {daily_dialog_len}")

print(f"Total Corpus Length: {len(corpus)}")


persona_tokens = sum(len(tokenizer.encode(t)) for t in corpus[:persona_len])
squad_tokens = sum(len(tokenizer.encode(t)) for t in corpus[persona_len:persona_len+squad_len])
daily_tokens = sum(len(tokenizer.encode(t)) for t in corpus[persona_len+squad_len:])
total = persona_tokens + squad_tokens + daily_tokens

print(f"{'Dataset':<20} {'Tokens':>12} {'%':>6}")
print("-" * 40)
print(f"{'Persona Chat':<20} {persona_tokens:>12,} {persona_tokens/total*100:>6.1f}%")
print(f"{'SQuAD':<20} {squad_tokens:>12,} {squad_tokens/total*100:>6.1f}%")
print(f"{'Daily Dialog':<20} {daily_tokens:>12,} {daily_tokens/total*100:>6.1f}%")
print("-" * 40)
print(f"{'Total':<20} {total:>12,}")



# flattening corpus & converting to tokens
tokens = []
for i in range(len(corpus)):
    tokens.extend(tokenizer.encode(corpus[i]))



X, y = [], []
for i in range(0, len(tokens) - block_size, block_size):
    X.append(tokens[i: i + block_size])
    y.append(tokens[i + 1: i + block_size + 1])

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

print(X.shape, y.shape)



class DecoderTransformerDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def __len__(self):
    return len(self.X)


# In[ ]:


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



import math

dataset = DecoderTransformerDataset(X, y)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size
)

model = DecoderTransformer().to(device)

# training scheduler 
total_steps = epochs * len(dataloader)
warmup_steps = int(0.01 * total_steps)

def lr_lambda(step): # cosine schedule
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

model.train()
for epoch in range(epochs):
  total_loss = 0.0
  pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
  for xb, yb in pbar:
    optimizer.zero_grad()

    xb = xb.to(device)
    yb = yb.to(device)

    out = model(xb)
    loss = criterion(out.transpose(1, 2), yb)
    total_loss += loss.item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
      
    pbar.set_postfix(loss=f"{loss.item():.4f}")
      
  print(f"Epoch: {epoch + 1}/{epochs} | Loss: {total_loss / len(dataloader)}")


# In[ ]:


test = "Hi how are you doing ? <sep> "
seq_len = 20

sep_token_id = tokenizer.convert_tokens_to_ids("<sep>")

def sample(logits, temperature=0.8):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
    
model.eval()
with torch.no_grad():
    input_tokens = tokenizer.encode(test)
    tokens = input_tokens.copy()
    
    for i, _ in enumerate(range(seq_len)):
        x = tokens[-block_size:]
        xb = torch.tensor(x).unsqueeze(0).to(device)
        out = model(xb)
        logits = out[0, -1]
        if i == 0:
            logits[sep_token_id] = float('-inf')  # suppress <sep> on first token only
        logits[tokenizer.eos_token_id] = float('-inf')
        next_token = sample(logits)
        if next_token == sep_token_id:  # stop at <sep>
            break
        tokens.append(next_token)

    print("Input:", test)
    print("Output:", tokenizer.decode(tokens[len(input_tokens):]))



torch.save(model.state_dict(), "/kaggle/working/model.pth")


