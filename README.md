# wikiGPT

A decoder-only GPT built from scratch in PyTorch, trained to hold conversations and answer questions. Inspired by the GPT-1 architecture (Radford et al., 2018).

---

## Model Weights & Notebook

The trained model weights and full training notebook are available on Kaggle:

- [Training Notebook](https://www.kaggle.com/code/gauravdharmadhikari/wikigpt)
- [Model Weights](https://www.kaggle.com/models/gauravdharmadhikari/wikigpt)
  
## Architecture

| Hyperparameter | Value |
|---|---|
| Layers (N) | 4 |
| Attention Heads (h) | 8 |
| Model Dimension (d_model) | 512 |
| Key/Value Dimension (d_k, d_v) | 64 |
| Context Length (block_size) | 128 |
| Dropout | 0.1 |

The model follows the standard transformer approach from (Vaswani et. al, 2017):
- Multi-head causal self-attention with masking
- Position-wise feedforward networks 
- Pre-norm residual connections (LayerNorm before each sublayer)
- Learned positional embeddings
- GPT-2 tokenizer with custom `<sep>` and `<eos>` special tokens

---

## Training

**Dataset:** ~9M tokens from three sources:

| Dataset | Tokens | % |
|---|---|---|
| PersonaChat | 3.5M | 39% |
| SQuAD v1.1 | 3.9M | 43% |
| DailyDialog | 1.6M | 18% |

**Format:**
- Conversational: `utterance <sep> response <sep> ... <eos>`
- QA: `context <sep> question <sep> answer <eos>`

**Optimizer:** Adam (lr=2.5e-4) with linear warmup (1% of steps) and cosine annealing to 0.

**Training:** 17 epochs on a Kaggle T4 GPU (~16 min/epoch). Best checkpoint saved at epoch 17 (loss: 2.57).

---

## Inference

```python
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"eos_token": "<eos>", "additional_special_tokens": ["<sep>"]})

model = DecoderTransformer().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
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
        for i in range(seq_len):
            x = tokens[-128:]
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

# Chat loop
history = []
print("Chatbot ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    history.append(user_input)
    prompt = " <sep> ".join(history[-6:])
    response = chat(prompt)
    history.append(response)
    print(f"Bot: {response}")
```

**Example output:**
```
You: How are you doing?
Bot: I'm doing ok, thank you. What do you like to accomplish for fun?

You: I like to work on coding projects
Bot: That is really cool! I wish that i could draw, but i like to draw comics.
```

---

## Files

| File | Description |
|---|---|
| `wikiGPT.ipynb` | Data preprocessing, model definition, and training |
| `wikiGPT_inference.ipynb` | Loading the model and running the chatbot |

---

## References

- Radford et al. (2018) — [Improving Language Understanding by Generative Pre-Training](https://openai.com/research/language-unsupervised)
- Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PersonaChat](https://arxiv.org/abs/1801.07243)
- [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)
- [DailyDialog](https://arxiv.org/abs/1710.04026)
