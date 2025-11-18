import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

st.set_page_config(page_title="Text Generator", layout="wide")
st.title("✨ AI Text Generator with Transformer")
st.markdown("Generate Shakespeare-like text using a trained transformer model")

# Sidebar config
st.sidebar.header("⚙️ Settings")
batch_size = st.sidebar.slider("Batch Size", 16, 256, 64)
block_size = st.sidebar.slider("Context Length", 16, 256, 128)
max_iters = st.sidebar.slider("Training Steps", 500, 10000, 3000, step=100)
eval_interval = st.sidebar.slider("Eval Every N Steps", 100, 1000, 300, step=50)
learning_rate = st.sidebar.number_input("Learning Rate", 1e-5, 0.01, 0.0001)
eval_iters = st.sidebar.slider("Eval Iterations", 10, 100, 50)

n_embd = st.sidebar.slider("Embedding Dim", 64, 512, 256)
n_embd = (n_embd // 8) * 8  # Make divisible by 8
n_head = st.sidebar.slider("Attention Heads", 4, 16, 8)
# Ensure embedding dim is divisible by heads
n_embd = ((n_embd // n_head) * n_head)
n_layer = st.sidebar.slider("Layers", 2, 12, 6)
dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.1)

max_new_tokens = st.sidebar.slider("Generate Tokens", 200, 2000, 500)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

st.sidebar.divider()
st.sidebar.info(f"🖥️ **Device:** {device.upper()}\n\n💡 **Tip:** Use higher settings for better quality (slower training)")

# Load data
try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    st.error("❌ Missing `input.txt`")
    st.stop()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Characters", f"{len(text):,}")
with col2:
    st.metric("Vocabulary", vocab_size)
with col3:
    st.metric("Device", device.upper())
with col4:
    st.metric("Model Size", f"{n_embd}→{n_layer}L")

st.divider()

# ============ TRANSFORMER MODEL ============
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'])
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size'])).view(1, 1, config['block_size'], config['block_size']))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.n_head)**0.5)
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config['n_embd'], 4 * config['n_embd'])
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config['n_embd'], config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']),
            wpe = nn.Embedding(config['block_size'], config['n_embd']),
            drop = nn.Dropout(config['dropout']),
            h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            ln_f = nn.LayerNorm(config['n_embd']),
        ))
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ============ TRAINING ============
st.header("🎓 Training")

if st.button("▶️ Start Training & Generate", use_container_width=True, key="train_btn"):
    try:
        # Setup
        torch.manual_seed(42)
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
        
        def get_batch(split):
            d = train_data if split == 'train' else val_data
            ix = torch.randint(len(d) - block_size, (batch_size,))
            x = torch.stack([d[i:i+block_size] for i in ix])
            y = torch.stack([d[i+1:i+block_size+1] for i in ix])
            return x.to(device), y.to(device)
        
        @torch.no_grad()
        def estimate_loss():
            out = {}
            model.eval()
            for split in ['train', 'val']:
                losses = []
                for _ in range(eval_iters):
                    x, y = get_batch(split)
                    _, loss = model(x, y)
                    losses.append(loss.item())
                out[split] = sum(losses) / len(losses)
            model.train()
            return out
        
        # Initialize
        config = {
            'vocab_size': vocab_size,
            'block_size': block_size,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'dropout': dropout,
        }
        
        st.info("📝 Initializing model...")
        model = GPT(config).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        st.info(f"✅ Model ready! **{total_params:,}** parameters")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
        
        # UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        
        train_losses = []
        val_losses = []
        steps = []
        
        st.info("🚀 Training in progress...")
        
        # Training loop
        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = estimate_loss()
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])
                steps.append(iter)
                
                pct = (iter + 1) / max_iters
                status_text.markdown(f"**Step {iter}/{max_iters}** | Train Loss: **{losses['train']:.4f}** | Val Loss: **{losses['val']:.4f}** | {int(pct*100)}%")
                progress_bar.progress(min(pct, 1.0))
                
                chart_data = pd.DataFrame({
                    'Step': steps,
                    'Train Loss': train_losses,
                    'Val Loss': val_losses
                })
                chart_placeholder.line_chart(chart_data.set_index('Step'), use_container_width=True)
            
            x, y = get_batch('train')
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        progress_bar.progress(1.0)
        st.success("✅ Training complete!")
        
        # Generate
        st.header("📝 Generated Text")
        st.info("🔄 Generating text with temperature=0.8...")
        
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        with torch.no_grad():
            generated_ids = model.generate(context, max_new_tokens=max_new_tokens, temperature=0.8)
        generated_text = decode(generated_ids[0].tolist())
        
        st.text_area("📖 Output:", generated_text, height=300, disabled=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download Text", generated_text, "generated.txt")
        with col2:
            if st.button("🎲 Generate Again (different output)"):
                with torch.no_grad():
                    context = torch.zeros((1, 1), dtype=torch.long, device=device)
                    new_ids = model.generate(context, max_new_tokens=max_new_tokens, temperature=0.9)
                st.text_area("New Output:", decode(new_ids[0].tolist()), height=300, disabled=True)
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())
