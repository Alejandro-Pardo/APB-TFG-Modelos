# NLP and Deep Learning for the Early Detection of Mental Disorders

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **Bachelor's Thesis (TFG)** — Alejandro Pardo Bascuñana  
> Universidad Carlos III de Madrid (UC3M)  
> Grado en Ingeniería Informática  (¡Computer Science and Engineering)
> Supervisor: Isabel Segura Bedmar — Madrid, 2024

> **Workshop Paper:** *[APB-UC3M at eRisk 2024](https://ceur-ws.org/Vol-3740/paper-82.pdf)* — CLEF 2024, CEUR-WS Vol. 3740  
> **Thesis URL:** *pending publication by Universidad Carlos III de Madrid*

---

## Overview

This repository contains the **model training logic, data analysis, and experimentation code** from the Bachelor's Thesis (*Trabajo de Fin de Grado*):

> **"Procesamiento de lenguaje natural y aprendizaje profundo para la detección temprana de enfermedades mentales"**  
> *(Natural Language Processing and Deep Learning for the Early Detection of Mental Disorders)*

> **Note:** The thesis document was written in Spanish.

The research was built upon the [CLEF eRisk 2024](https://erisk.irlab.org/2024/index.html) shared task on **Early Risk Prediction on the Internet**, where the team **APB-UC3M** participated in Tasks 1 and 2. The competition-specific submission code and working notes will live in a separate repository — this repo focuses exclusively on the model development and analysis described in the thesis.

---

## Context: eRisk 2024 Shared Task

[eRisk](https://erisk.irlab.org/2024/index.html) explores evaluation methodology, effectiveness metrics, and practical applications of **early risk detection on the Internet**, with a focus on health and safety. It is organized as a lab at the annual [CLEF conference](http://clef2024.clef-initiative.eu/).

The thesis addresses **two of the three tasks** proposed in 2024:

| Task | Description | Goal |
|------|-------------|------|
| **Task 1** | Search for Symptoms of Depression | Rank sentences from user writings by their relevance to each of the 21 symptoms in the BDI-II questionnaire |
| **Task 2** | Early Detection of Signs of Anorexia | Sequentially process user writings and detect early traces of anorexia as soon as possible |

---

## Repository Structure

```
├── README.md
├── task1_depression_symptoms/
│   ├── bdi-ii.txt                       # BDI-II questionnaire items (21 symptoms)
│   ├── 01_eda.ipynb                     # Exploratory Data Analysis
│   ├── 02_classifier_training.ipynb     # Fine-tuning RoBERTa (go_emotions) for 21-class classification
│   ├── 03_sentence_similarity.ipynb     # Sentence similarity approach (MiniLM, MPNet)
│   ├── 04_inference.ipynb               # Inference & evaluation with the trained classifier
│   └── 05_results_formatting.ipynb      # Post-processing into TREC submission format
│
└── task2_anorexia_detection/
    ├── 01_eda.ipynb                     # Exploratory Data Analysis (eRisk 2018/2019 data)
    ├── 02_doc2vec_ensemble.ipynb        # Doc2Vec embeddings + SVM/LR/RF ensemble
    ├── 03_transformer_ensemble.ipynb    # BART-base embeddings + SVM/LR/RF ensemble
    └── server_client.py                 # Client for the eRisk real-time evaluation server
```

---

## Workflow

### Task 1 — Search for Symptoms of Depression

1. **EDA** ([01_eda.ipynb](task1_depression_symptoms/01_eda.ipynb)): Analyze the training corpus (XML documents + relevance judgements). Study label distributions, text lengths, and per-symptom frequencies.  
2. **Classifier Training** ([02_classifier_training.ipynb](task1_depression_symptoms/02_classifier_training.ipynb)): Fine-tune a custom 21-class classification head on top of `SamLowe/roberta-base-go_emotions` using HuggingFace Transformers. Optimized with AdamW + linear LR schedule over 3 epochs.  
3. **Sentence Similarity** ([03_sentence_similarity.ipynb](task1_depression_symptoms/03_sentence_similarity.ipynb)): Alternative approach using cosine similarity between sentence embeddings (Sentence-Transformers: `all-MiniLM-L6-v2`, `all-MiniLM-L12-v2`, `all-mpnet-base-v2`) and per-symptom training clusters.  
4. **Inference** ([04_inference.ipynb](task1_depression_symptoms/04_inference.ipynb)): Load the trained RoBERTa classifier, predict labels and relevance scores on test data, and compute evaluation metrics (micro F1, precision, recall).  
5. **Results Formatting** ([05_results_formatting.ipynb](task1_depression_symptoms/05_results_formatting.ipynb)): Aggregate predictions, select top-1000 per symptom, and format output as TREC-style submission files for all runs (including an ensemble).

### Task 2 — Early Detection of Signs of Anorexia

1. **EDA** ([01_eda.ipynb](task2_anorexia_detection/01_eda.ipynb)): Parse XML data from eRisk 2018 and 2019 training sets, assign golden-truth labels (anorexia vs. control), analyze text length distributions.  
2. **Doc2Vec Ensemble** ([02_doc2vec_ensemble.ipynb](task2_anorexia_detection/02_doc2vec_ensemble.ipynb)): Train a Doc2Vec model (vector_size=100, 30 epochs), then use document vectors as features for an ensemble (soft-voting) of SVM, Logistic Regression, and Random Forest — each tuned via GridSearchCV with StratifiedKFold.  
3. **Transformer Ensemble** ([03_transformer_ensemble.ipynb](task2_anorexia_detection/03_transformer_ensemble.ipynb)): Same ensemble approach but using `facebook/bart-base` sentence embeddings (via Sentence-Transformers) instead of Doc2Vec.  
4. **Server Client** ([server_client.py](task2_anorexia_detection/server_client.py)): Client that connects to the eRisk evaluation server, iteratively receives user writings, processes text (lemmatization, stopword removal via spaCy), generates predictions with both ensemble models, and submits decisions in real-time.

---

## Key Technologies

- **Transformers**: HuggingFace Transformers, Sentence-Transformers
- **Models**: `SamLowe/roberta-base-go_emotions`, `facebook/bart-base`, `all-MiniLM-L6-v2`, `all-MiniLM-L12-v2`, `all-mpnet-base-v2`
- **ML**: scikit-learn (SVM, Logistic Regression, Random Forest, VotingClassifier, GridSearchCV)
- **Embeddings**: Doc2Vec (Gensim), Sentence-Transformers
- **NLP**: spaCy (`en_core_web_sm`)
- **Deep Learning**: PyTorch

---

## Citation

If you use this work, please cite the thesis:

```bibtex
@thesis{pardo2024tfg,
  title     = {{Procesamiento de lenguaje natural y aprendizaje profundo para la detección temprana de enfermedades mentales}},
  author    = {Pardo Bascu{\~n}ana, Alejandro},
  school    = {Universidad Carlos III de Madrid},
  year      = {2024},
  type      = {Bachelor's Thesis (Trabajo de Fin de Grado)},
  note      = {Supervised by Isabel Segura Bedmar},
  url       = {}
}
```

> **Thesis URL:** *pending publication by Universidad Carlos III de Madrid — will be added once available.*

---

## Related Publication

The competition working notes paper derived from this thesis:

- Alejandro Pardo Bascuñana, Isabel Segura Bedmar. *"APB-UC3M at eRisk 2024: Natural Language Processing and Deep Learning for the Early Detection of Mental Disorders"*. Working Notes of CLEF 2024 (CEUR-WS Vol. 3740). [PDF](https://ceur-ws.org/Vol-3740/paper-82.pdf)

---

## Links

- **eRisk 2024 Official Page:** [https://erisk.irlab.org/2024/index.html](https://erisk.irlab.org/2024/index.html)
- **CLEF 2024 Conference:** [http://clef2024.clef-initiative.eu/](http://clef2024.clef-initiative.eu/)
- **Workshop Paper (CEUR-WS):** [https://ceur-ws.org/Vol-3740/paper-82.pdf](https://ceur-ws.org/Vol-3740/paper-82.pdf)

---

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

## Acknowledgments

To my family, my supervisor Isabel, my teammate Pedro, and all the professors at UC3M who made learning about AI a constant discovery.

---

*Alejandro Pardo Bascuñana — Universidad Carlos III de Madrid, 2024*
