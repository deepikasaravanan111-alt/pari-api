# GPARI — Genetic-Psychological Addiction Risk Index

**Does COMT Val158Met, Not DRD2, Drive Compulsive Risk-Taking?**
A Machine Learning and Causal Mediation Analysis of Dopaminergic Pathways to Sensation-Seeking Addiction

**Deepika Saravanan** — John P. Stevens High School, Edison, NJ, USA
deepika.saravanan111@gmail.com

---

## Overview

This repository contains the training code, trained model, and backend API for the GPARI screening tool. GPARI is a gradient boosting classifier trained on Add Health Wave IV public use data (n = 6,504) that predicts compulsive risk-taking outcomes and identifies whether an individual's risk runs through COMT-linked inhibitory control failure or DRD2-linked reward insensitivity.

The central finding: COMT Val158Met endophenotypes outpredict DRD2 Taq1A endophenotypes in a compulsive risk-taking outcome model (AUC = 0.885), with causal mediation confirming inhibitory control failure carries 67% of the total indirect COMT effect (95% CI: 54% to 79%, p < 0.001).

**Live tool:** https://deepikasaravanan111-alt.github.io/pari
**Preprint:** [link when available]

---

## Repository Structure
pari-api/
├── train.py              # Training script
├── main.py               # FastAPI backend
├── gpari_model.joblib    # Trained model
├── gpari_data.joblib     # Train/test split
├── requirements.txt      # Dependencies
└── README.md             # This file

---

## Data

**Dataset:** Add Health Wave IV Public Use File
**Source:** ICPSR Study 21600
**Access:** Free, no institutional approval required
**URL:** https://www.icpsr.umich.edu/web/DSDR/studies/21600
**Sample:** n = 6,504, mean age 28.9

| Gene | Mechanism | Endophenotype | Add Health Code |
|------|-----------|---------------|-----------------|
| COMT Val158Met | Prefrontal dopamine; stress-contingent inhibitory failure | Stress-load impulse control | H4MH22 |
| DRD2 Taq1A | Striatal D2 density; reward insensitivity | Reward responsiveness | H4ID5G |
| SLC6A4 | Serotonin reuptake; anxiety sensitivity | Trait anxiety | H4MH2 |
| MAOA | Monoamine metabolism; emotional reactivity | Emotional dysregulation | H4MH7 |
| General | Motivational deficit | Anhedonia | H4MH19 |
| General | Behavioral engagement | Novelty-seeking composite | H4FS1 |
| General | Adversity moderator | ACE composite | H4MH23 |

---

## Model

**Algorithm:** sklearn GradientBoostingClassifier
**Hyperparameters:** n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42

| Model | AUC |
|-------|-----|
| Gradient Boosting | 0.885 +/- 0.010 |
| XGBoost | 0.879 |
| Random Forest | 0.861 |
| Logistic Regression (baseline) | 0.731 |

---

## Reproducing the Results

```bash
pip install -r requirements.txt
python train.py
uvicorn main:app --host 0.0.0.0 --port 8000
```

Send a POST request to /predict:

```json
{
  "inhibitory_control": 4.0,
  "reward_sensitivity": 2.0,
  "trait_anxiety": 3.5,
  "emotional_dysreg": 3.0,
  "anhedonia": 2.5,
  "novelty_seeking": 4.0,
  "ace_score": 2.0,
  "age": 16.0,
  "income": 3.0
}
```

---

## Live Deployment

- **Frontend:** https://deepikasaravanan111-alt.github.io/pari
- **Backend API:** https://pari-api-production.up.railway.app
- **Health check:** https://pari-api-production.up.railway.app/health

---

## Key Results

- COMT proxy (H4MH22) outranks DRD2 proxy (H4ID5G) by 1.8x in male-stratified model
- Indirect COMT effect through inhibitory control failure: 67% (95% CI: 54% to 79%, p < 0.001)
- Male vs. female feature importance correlation: r = 0.58

---

## Citation

Saravanan, D. (2026). Does COMT Val158Met, Not DRD2, Drive Compulsive Risk-Taking? A Machine Learning and Causal Mediation Analysis of Dopaminergic Pathways to Sensation-Seeking Addiction. [Preprint]

---

## License

MIT License. The underlying Add Health data is subject to ICPSR terms of use.

---

## Contact

Deepika Saravanan
deepika.saravanan111@gmail.com
John P. Stevens High School, Edison, NJ, USA
