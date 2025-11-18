import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

# Page configuration
st.set_page_config(page_title="Simple Language Model", layout="wide")
st.title("🤖 Simple Character-Level Language Model")
st.markdown("Train and generate text using a neural network model")

# Hyperparameters in sidebar
st.sidebar.header("⚙️ Hyperparameters")
batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)
block_size = st.sidebar.slider("Block Size (Context)", 4, 64, 16)
max_iters = st.sidebar.slider("Max Iterations", 100, 5000, 1000, step=100)
eval_interval = st.sidebar.slider("Eval Interval", 50, 500, 100, step=50)
learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 0.1, 0.01, format="%.5f")
eval_iters = st.sidebar.slider("Eval Iterations", 20, 200, 50, step=10)
max_new_tokens = st.sidebar.slider("Max New Tokens", 100, 1000, 300)
hidden_size = st.sidebar.slider("Hidden Size", 64, 512, 128)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.info(f"📍 Device: {device}")

# Load text
try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    st.error("❌ `input.txt` file not found!")
    st.stop()

# Create character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Display stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Text Length", f"{len(text):,}")
with col2:
    st.metric("Vocabulary Size", vocab_size)
with col3:
    st.metric("Device", device.upper())

st.divider()

# Simple model
class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, block_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        x = self.embedding(idx)  # (B, T, hidden_size)
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_size)
        logits = self.fc(lstm_out)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
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
        
        # Initialize model
        st.info("📝 Initializing model...")
        model = SimpleLanguageModel(vocab_size, hidden_size, block_size)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        
        train_losses = []
        val_losses = []
        steps = []
        
        st.info("🚀 Training started...")
        
        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = estimate_loss()
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])
                steps.append(iter)
                
                status_text.write(f"Step {iter}/{max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
                progress_bar.progress(min((iter + 1) / max_iters, 1.0))
                
                chart_data = pd.DataFrame({
                    'Step': steps,
                    'Train Loss': train_losses,
                    'Val Loss': val_losses
                })
                chart_placeholder.line_chart(chart_data.set_index('Step'))
            
            x, y = get_batch('train')
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        progress_bar.progress(1.0)
        st.success("✅ Training completed!")
        
        # Generate text
        st.header("📝 Generated Text")
        with st.spinner("🔄 Generating text..."):
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated_text = decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())
        
        st.text_area("Output:", generated_text, height=250, disabled=True)
        
        st.download_button(
            label="📥 Download Text",
            data=generated_text,
            file_name="generated_text.txt"
        )
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error(str(type(e).__name__))
