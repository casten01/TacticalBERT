# TacticalBERT: Decoding Football Tactics with Transformers

## Overview
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

### Project Structure

The repository is organized into the following key directories and files:

* **`analysis/`** contains Jupyter Notebooks dedicated to exploratory data analysis and model interpretability. You will find the analyses on attention mechanisms (`attention_analysis.ipynb`), events embeddings analysis  (`geometric_tactical_events.ipynb`), and player embeddings (`players_semantic_atlas.ipynb`).
  
* **`models/`** stores the pre-trained weights of the model (`model_best.pth`, re-trained with complete dataset `model_production.pth`), along with the training state logs and the final classification reports (`best_report.txt`).

* **`src/`** the core source code of the project. It includes the custom architecture (`model.py`), the data loading and collating logic (`dataset.py`), the tactical vocabulary builder (`tokenizer.py`), and the global hyperparameters (`config.py`).

* **`Root Directory`** contains the main executable scripts that form the pipeline (`extract_data.py`, `process_data.py`, `generate_events_vectors.py`, `generate_players_vectors.py`, and `train.py`), along with environment setups (`requirements.txt`).

* **`data/`** *(Generated Locally)* this directory is created automatically when running the extraction scripts. It houses the `raw/` downloaded Parquet chunks and the `processed/` tokenized PyTorch datasets.

## Getting Started

To reproduce the analysis or train the model from scratch, please follow the setup instructions carefully.

### 1. Prerequisites & Virtual Environment
It is highly recommended to use a virtual environment to avoid dependency conflicts. 

```bash
# Clone the repository
git clone https://github.com/casten01/TacticalBERT.git
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
```bash
pip install -r requirements.txt
```
### 3. Data Pipeline Execution

To process the raw StatsBomb data and generate the contextual embeddings required by the model, you must run the following data preparation scripts in this exact order:

- `extract_data.py`: Connects to the data source/API and downloads the raw JSON event files for the selected matches.

- `process_data.py`: Cleans the raw data, handles missing values, and structures the events into continuous possession sequences.

- `generate_events_vectors.py`: Converts the categorical tactical events (e.g., Pass, Carry) into numerical tokens based on the custom type_vocab.

- `generate_players_vectors.py`: Creates contextual embeddings for the players involved, mapping their roles and physical coordinates on the pitch.

```bash
python extract_data.py
python process_data.py
python generate_events_vectors.py
python generate_players_vectors.py
```
### 4. Model Weights

The pre-trained TacticalBERT model weights are already included in the `model` directory. You do not need to retrain the model to run the interpretability notebooks.

You can retrain the model from scratch using the `train.py` script. The training logic is designed as a two-phase pipeline:

* **Validation & Early Stopping:** the script automatically performs an 80/20 dataset split. It trains the model while monitoring the Validation F1-Score (Macro), applying a custom class-weight dictionary to penalize errors on rare/crucial events. It uses Early Stopping (max_epochs=100 and patience=15) to identify the best training epoch and saves a detailed classification report `models/best_report.txt` .

* **Full Production Retraining:** once the optimal number of epochs is found, the script automatically re-initializes and retrains the model on **100% of the dataset** for exactly that number of epochs, maximizing data exploitation.

To start the training process, simply run:
```bash
python train.py
```


## Interpretability & Notebooks

After successfully running the data pipeline, you can explore the Jupyter Notebooks. These notebooks contain the core analytical work, combining theoretical NLP research with sports analytics.

## Generative AI Disclaimer

I use generative AI during the development of this research project.

* **Models Used:** Google Gemini
* **Purposes:** AI was utilized as an interactive research assistant to help brainstorm and summarize ideas and assist in drafting Python code.

All AI-generated outputs were treated strictly as drafts. Every piece of code was reviewed, tested, and  modified. All final conclusions and analyses are entirely the author's original work.

## Inspiration & Data Sources

This project bridges the gap between natural language processing and sports analytics, relying on high-quality open data and foundational research in transformer interpretability.

### Data Source
All tactical sequences, physical coordinates, and event metadata used to train and evaluate TacticalBERT are sourced from the **[StatsBomb Open Data Repository](https://github.com/statsbomb/open-data)**. The dataset considers the comprehensive event data from the 2015/2016 season across the "Big Five" European leagues.

### Academic Inspiration

* **Transformer Interpretability & Geometry:**
  * **Reif, E., et al. (2019).** *Visualizing and measuring the geometry of BERT.* This work inspired approach to projecting the model's latent space, demonstrating that a transformer's embeddings possess a searchable geometry where semantic (and in our case, tactical) relationships are encoded as spatial distances.
  * **Abnar, S., & Zuidema, W. (2020).** *Quantifying attention flow in transformers.* This paper guided the diagnostic approach to multi-head attention, highlighting the "Raw Attention Problem" and the necessity of tracking token mixing across layers to find the true causal dependencies in a sequence.

* **Language Models in Football Analytics:**
  * **Adjileye, A. A. (2024).** *RisingBALLER: A player is a token, a match is a sentence...* This research pioneered the paradigm of treating football data strictly as language, providing the foundational concept that sequential player involvements can be modeled exactly like textual sentences.
  * **Hong, M., et al. (2025).** *ScoutGPT: Capturing Player Impact from Team Action Sequences Using GPT-Based Framework.* This recent work reinforced the validity of using sequence-based generative frameworks to isolate and understand complex player impacts and tactical behaviors within team actions.



