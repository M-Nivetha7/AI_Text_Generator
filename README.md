# 🧠 AI Text Generator  
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)  
![AI](https://img.shields.io/badge/Generative-AI-green.svg)  
![License](https://img.shields.io/badge/License-MIT-purple.svg)

A powerful **AI Text Generator** that blends **Bigram-based NLP** with a **GPT-style neural generator**, wrapped inside an interactive **Streamlit Web App**.  
Perfect for exploring how classical models compare to modern LLM-style text generation.  

---

## 🚀 Features  
- 🔤 **Bigram Text Generator** – Statistical & simple predictive model  
- 🤖 **GPT-inspired Generator** – Neural-based, coherent text outputs  
- 🌐 **Interactive Streamlit UI** – Generate text in seconds  
- 📁 **Input / Output Logging** – Reads `input.txt`, writes to `output.log`  
- 🧩 Modular, beginner-friendly Python code  

---

## 📂 Directory Structure  
AI_Text_Generator/
│── bigram.py # Bigram model
│── gpt.py # GPT-style generator
│── streamlit_app.py # Streamlit UI
│── testing_ui.py # UI testing module
│── input.txt # Sample training data
│── output.log # Generated text logs
│── README.md # Documentation

yaml
Copy code

---

## 🛠 Installation  

### 1️⃣ Clone the Repo  
```bash
git clone https://github.com/M-Nivetha7/AI_Text_Generator.git
cd AI_Text_Generator
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the Streamlit App
bash
Copy code
streamlit run streamlit_app.py
🧪 How to Use
✍️ Enter your prompt

⚙️ Choose Bigram or GPT

🚀 Press Generate

📄 View the generated output

📘 Check output.log for saved results

🧠 How It Works
🔹 Bigram Model
A probability-based model that predicts the next word using word pairs.
Simple, lightweight & helps understand traditional NLP.

🔹 GPT-based Generator
A neural model that generates text by learning deeper patterns.
More fluent, contextual & human-like.

✨ Learning Outcomes
Difference between classical NLP & deep learning text generators

Tokenization, probability modeling, and log-likelihood

End-to-end AI pipeline development

Building & deploying interactive ML apps with Streamlit

Logging, debugging, and evaluating generated text

🌟 Demo (Optional)
Add a demo GIF or screenshot here:

scss
Copy code
![Demo Screenshot](demo.png)
🤝 Contributing
PRs, issues, and suggestions are always welcome!
Feel free to enhance the UI, models, or documentation.

📬 Contact
👩‍💻 Author: M. Nivetha
