import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

# Page configuration
st.set_page_config(page_title="Advanced Language Model", layout="wide")
st.title("🤖 Advanced Language Model with Attention")
st.markdown("Train and generate meaningful text using a transformer-based model")

# Hyperparameters in sidebar
st.sidebar.header("Hyperparameters")
batch_size = st.sidebar.slider("Batch Size", 8, 128, 64)
block_size = st.sidebar.slider("Block Size (Context Length)", 8, 128, 64)
max_iters = st.sidebar.slider("Max Iterations", 100, 10000, 5000, step=100)
eval_interval = st.sidebar.slider("Eval Interval", 50, 500, 300, step=50)
learning_rate = st.sidebar.number_input("Learning Rate", 0.00001, 0.1, 0.001, format="%.5f")
eval_iters = st.sidebar.slider("Eval Iterations", 50, 500, 200, step=50)
max_new_tokens = st.sidebar.slider("Max New Tokens to Generate", 100, 2000, 500)

# Model architecture
n_embd = st.sidebar.slider("Embedding Dimension", 64, 512, 256)
n_head = st.sidebar.slider("Number of Attention Heads", 1, 16, 4)
n_layer = st.sidebar.slider("Number of Layers", 1, 8, 3)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.info(f"📍 Device: {device}")

# Load text
try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    st.error("❌ `input.txt` file not found. Please ensure it's in the same directory.")
    st.stop()

# Create character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Display text info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Text Length", f"{len(text):,} characters")
with col2:
    st.metric("Unique Characters", vocab_size)
with col3:
    st.metric("Device", device.upper())

st.divider()

# Advanced Model with Multi-Head Attention
class Head(nn.Module):
    """Single head of self-attention"""
    def __init__(self, head_size, block_size, n_embd):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, block_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size, n_embd) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, block_size, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Training section
st.header("🎓 Training")

if st.button("▶️ Start Training", use_container_width=True):
    try:
        torch.manual_seed(1337)
        
        # Prepare data
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
        
        def get_batch(split):
            data = train_data if split == 'train' else val_data
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([data[i:i+block_size] for i in ix])
            y = torch.stack([data[i+1:i+block_size+1] for i in ix])
            x, y = x.to(device), y.to(device)
            return x, y
        
        @torch.no_grad()
        def estimate_loss():
            out = {}
            model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    X, Y = get_batch(split)
                    logits, loss = model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            model.train()
            return out
        
        # Initialize model
        st.info("📝 Initializing model...")
        model = LanguageModel(vocab_size, n_embd, n_head, n_layer, block_size)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        
        train_losses = []
        val_losses = []
        steps = []
        
        st.info("🚀 Training started...")
        
        # Training loop
        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = estimate_loss()
                train_losses.append(losses['train'].item())
                val_losses.append(losses['val'].item())
                steps.append(iter)
                
                status_text.write(f"Step {iter}/{max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
                progress_bar.progress((iter + 1) / max_iters)
                
                # Update chart
                chart_data = pd.DataFrame({
                    'Step': steps,
                    'Train Loss': train_losses,
                    'Val Loss': val_losses
                })
                chart_placeholder.line_chart(chart_data.set_index('Step'))
            
            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        st.success("✅ Training completed!")
        
        # Generate text
        st.header("📝 Generate Text")
        st.info("🔄 Generating text...")
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_text = decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())
        
        st.text_area("Generated Text:", generated_text, height=300, disabled=True)
        
        # Download button
        st.download_button(
            label="📥 Download Generated Text",
            data=generated_text,
            file_name="generated_text.txt",
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"❌ Error during training: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    st.success("✅ Training completed!")
    
    # Generate text
    st.header("📝 Generate Text")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())
    
    st.text_area("Generated Text:", generated_text, height=300, disabled=True)
    
    # Download button
    st.download_button(
        label="📥 Download Generated Text",
        data=generated_text,
        file_name="generated_text.txt",
        mime="text/plain"
    )
