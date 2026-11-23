import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        block_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.blocks(x, mask=causal_mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int = 220, temperature: float = 0.8):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def train_model(
    text_path: str = "sample_stories.txt",
    ckpt_path: str = "tiny_transformer.pt",
    steps: int = 400,
    lr: float = 3e-3,
    block_size: int = 96,
):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        block_size=block_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    model.train()
    for _ in range(steps):
        if len(data) <= block_size + 1:
            break
        ix = torch.randint(0, len(data) - block_size - 1, (16,))
        x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)

        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    torch.save(
        {
            "model_state": model.state_dict(),
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "block_size": block_size,
            "losses": losses,
        },
        ckpt_path,
    )
    return losses


def load_checkpoint(ckpt_path: str = "tiny_transformer.pt"):
    if not os.path.exists(ckpt_path):
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    tokenizer = CharTokenizer("")
    tokenizer.stoi = ckpt["stoi"]
    tokenizer.itos = ckpt["itos"]
    tokenizer.vocab_size = len(ckpt["stoi"])

    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        block_size=ckpt["block_size"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, tokenizer, ckpt.get("losses", []), device


@st.cache_resource
def load_creative_transformer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id)
    mdl.eval()
    return tok, mdl


def generate_creative_story(prompt: str, max_new_tokens: int = 160, temperature: float = 0.85) -> str:
    tok, mdl = load_creative_transformer()
    messages = [
        {
            "role": "system",
            "content": "You are a suspenseful horror and mystery storyteller. Write a vivid, gripping short story (3 to 4 complete sentences) continuing the user's idea with eerie atmosphere and a twist.",
        },
        {"role": "user", "content": prompt},
    ]
    input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(temperature, 0.2),
            top_p=0.92,
            repetition_penalty=1.15,
            pad_token_id=tok.eos_token_id,
        )
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    story = tok.decode(generated_tokens, skip_special_tokens=True).strip()
    return story
