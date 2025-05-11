# AI Travel Planner with Streamlit

This project is a Streamlit-based web application that uses multiple AI agents to recommend:

- The best months to visit a location (weather-based).
- Personalized hotel recommendations (via sentence embeddings).
- A complete day-by-day travel itinerary (via OpenAI GPT-4).

---

## How to Run the App

### 1. Clone or Download the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and Activate a Virtual Environment by using requirements.txt attached

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 3. Install the Requirements

```bash
pip install -r requirements.txt
```

### 4. Set up Streamlit Secrets

Create a `.streamlit/secrets.toml` file and insert your OpenAI API key:
Example of the content:

```toml
[general]
openai_api_key = "xxx_yyy_zzz..."
```

> 💡 Make sure this file is in the root of your project inside the `.streamlit` folder.

---

### 5. Run the App

```bash
streamlit run Multi_Model–Travel_Planning_System_streamlit_v_0_2.py
```

Then open the app at:

- http://localhost:8501

---

## ⚠️ Known Warnings & Fixes

### Tokenizers Parallelism Warning

Add this line at the top of your script to prevent HuggingFace fork-related deadlocks:

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### Torch Custom Class RuntimeError

You might see:

```
RuntimeError: Tried to instantiate class '__path__._path'...
```

This is a harmless warning caused by Streamlit's file watcher trying to inspect PyTorch internals. It can be ignored.

---

## 🛠 Requirements

Key Python packages used:

- `streamlit`
- `openai`
- `sentence-transformers`
- `scikit-learn`
- `pydeck`
- `pandas`, `numpy`

---

## 📄 License

MIT License