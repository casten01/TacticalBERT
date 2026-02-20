# TacticalBERT: Decoding Football Tactics with Transformers

## 📖 Overview
This repository contains the code and implementation for **TacticalBERT**, a Transformer-based model designed to analyze football (soccer) event data. Drawing inspiration from Natural Language Processing (NLP) literature, this project treats football matches as a language, where events (passes, carries, duels) are words, and ball possessions are sentences. 

The input representation of the model is a composite embedding constructed by summing the standard base tokens with custom tactical dimensions: the discretized spatial location on the pitch, the event's duration, and a linear projection of continuous contextual features (such as defensive pressure).

### Architecture Configuration
* **Model:** custom BERT
* **Layers (Depth):** 4
* **Attention Heads:** 4
* **Embedding Dimension (Hidden Size):** 128
* **Maximum Sequence Length:** 128

### Training Hyperparameters
The model was trained from scratch with:
* **Optimizer:** AdamW (`lr=1e-4`)
* **Loss Function:** Weighted Cross-Entropy (custom weights applied to heavily penalize errors on rare, high-value events)
* **Batch Size:** 32
* **Regularization:** Early Stopping with a patience of 15 epochs, monitoring the Validation Macro F1-Score.

## 🚀 Getting Started

To reproduce the analysis or train the model from scratch, please follow the setup instructions carefully.

### 1. Prerequisites & Virtual Environment
It is highly recommended to use a virtual environment to avoid dependency conflicts. 

```bash
# Clone the repository
git clone [https://github.com/yourusername/TacticalBERT.git](https://github.com/yourusername/TacticalBERT.git)
cd TacticalBERT

# Create a virtual environment (Python 3.8+ recommended)
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Once the virtual environment is active, install the required packages:
```Bash
pip install -r requirements.txt
```

### 3. Model Weights

The pre-trained TacticalBERT model weights are already included in the `model` directory. You do not need to retrain the model to run the interpretability notebooks.

You can retrain the model from scratch using the `train.py` script. The training logic is designed as a two-phase pipeline:

* **Validation & Early Stopping:** the script automatically performs an 80/20 dataset split. It trains the model while monitoring the Validation F1-Score (Macro), applying a custom class-weight dictionary to penalize errors on rare/crucial events. It uses Early Stopping (max_epochs=100 and patience=15) to identify the best training epoch and saves a detailed classification report `models/best_report.txt` .

* **Full Production Retraining:** once the optimal number of epochs is found, the script automatically re-initializes and retrains the model on **100% of the dataset** for exactly that number of epochs, maximizing data exploitation.

To start the training process, simply run:
```bash
python train.py
```

## ⚙️ Data Pipeline Execution

To process the raw StatsBomb data and generate the contextual embeddings required by the model, you must run the following data preparation scripts in this exact order:

- `extract_data.py`: Connects to the data source/API and downloads the raw JSON event files for the selected matches.

- `process_data.py`: Cleans the raw data, handles missing values, and structures the events into continuous possession sequences.

- `generate_events_vectors.py`: Converts the categorical tactical events (e.g., Pass, Carry) into numerical tokens based on the custom type_vocab.

- `generate_players_vectors.py`: Creates contextual embeddings for the players involved, mapping their roles and physical coordinates on the pitch.

Bash
```
python extract_data.py
python process_data.py
python generate_events_vectors.py
python generate_players_vectors.py
```

## 📊 Interpretability & Notebooks

After successfully running the data pipeline, you can explore the Jupyter Notebooks. These notebooks contain the core analytical work, combining theoretical NLP research with sports analytics.

## 🤖 Generative AI Disclaimer

I use of generative AI during the development of this research project.

* **Models Used:** Google Gemini
* **Purposes:** AI was utilized as an interactive research assistant to help brainstorm and summarize ideas and assist in drafting Python code.

All AI-generated outputs were treated strictly as drafts. Every piece of code was reviewed, tested, and  modified. All final conclusions and analyses are entirely the author's original work.


