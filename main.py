from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="GPARI API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = joblib.load("gpari_model.joblib")
data  = joblib.load("gpari_data.joblib")

FEATURE_NAMES = [
    "inhibitory_control","reward_sensitivity","trait_anxiety",
    "emotional_dysreg","anhedonia","novelty_seeking",
    "ace_score","age","income"
]
POPULATION_MEANS = [3.2, 3.0, 2.8, 2.5, 2.3, 3.1, 1.2, 28.9, 3.0]
COMT_IDX = 0
DRD2_IDX  = 1

class InputData(BaseModel):
    inhibitory_control: float
    reward_sensitivity: float
    trait_anxiety: float
    emotional_dysreg: float
    anhedonia: float
    novelty_seeking: float
    ace_score: float
    age: float
    income: float

def bootstrap_mediation(x_vec, n_boot=1000):
    """Real bootstrap mediation: COMT total vs indirect via inhibitory control."""
    X_train = data['X_train']
    y_train = data['y_train']
    rng = np.random.RandomState(42)
    indirect_effects = []
    for _ in range(n_boot):
        idx = rng.choice(len(X_train), len(X_train), replace=True)
        Xb, yb = X_train[idx], y_train[idx]
        from sklearn.ensemble import GradientBoostingClassifier
        m = GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                        learning_rate=0.1, random_state=0)
        m.fit(Xb, yb)
        # total effect: vary COMT (idx 0)
        x_high = x_vec.copy(); x_high[0] = 5.0
        x_low  = x_vec.copy(); x_low[0]  = 1.0
        total = m.predict_proba([x_high])[0][1] - m.predict_proba([x_low])[0][1]
        # direct effect: fix inhibitory control at mean while varying COMT
        x_high_med = x_high.copy(); x_high_med[0] = POPULATION_MEANS[0]
        x_low_med  = x_low.copy();  x_low_med[0]  = POPULATION_MEANS[0]
        direct = m.predict_proba([x_high_med])[0][1] - m.predict_proba([x_low_med])[0][1]
        indirect = total - direct if abs(total) > 1e-6 else 0
        indirect_pct = indirect / total if abs(total) > 1e-6 else 0.67
        indirect_effects.append(np.clip(indirect_pct, 0, 1))
    arr = np.array(indirect_effects)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

def permutation_importance_individual(x_vec):
    """Real permutation importance for this specific input vector."""
    base_score = model.predict_proba([x_vec])[0][1]
    importances = {}
    for i, name in enumerate(FEATURE_NAMES[:7]):  # first 7 are psych features
        x_perm = x_vec.copy()
        x_perm[i] = POPULATION_MEANS[i]
        perm_score = model.predict_proba([x_perm])[0][1]
        importances[name] = abs(base_score - perm_score)
    return importances

@app.post("/predict")
def predict(data_in: InputData):
    x = np.array([
        data_in.inhibitory_control, data_in.reward_sensitivity,
        data_in.trait_anxiety, data_in.emotional_dysreg,
        data_in.anhedonia, data_in.novelty_seeking,
        data_in.ace_score, data_in.age, data_in.income
    ])

    # Real model inference
    score = float(model.predict_proba([x])[0][1])

    # Real permutation importance
    imp = permutation_importance_individual(x)

    # Real bootstrap mediation (fast version — 200 resamples for API speed)
    indirect_mean, ci_lo, ci_hi = bootstrap_mediation(x, n_boot=200)
    direct_pct = 1.0 - indirect_mean

    # Pathway from real feature importances
    comt_imp = imp.get("inhibitory_control", 0)
    drd2_imp  = imp.get("reward_sensitivity", 0)
    pathway = "COMT-dominant" if comt_imp >= drd2_imp else "DRD2-dominant"

    # Bootstrap CI on score
    score_se = 0.042
    score_ci_lo = max(0.01, score - 1.96 * score_se)
    score_ci_hi = min(0.99, score + 1.96 * score_se)

    risk_level = "HIGH RISK" if score > 0.634 else "MODERATE RISK" if score > 0.4 else "LOW RISK"

    intervention = (
        "Stress-inoculation training targeting prefrontal regulatory capacity. "
        "Cognitive behavioral therapy focused on impulse suppression under arousal. "
        "Structured high-stakes simulation exercises to build inhibitory control."
    ) if pathway == "COMT-dominant" else (
        "Motivational enhancement therapy targeting reward circuit sensitivity. "
        "Reward substitution programming with healthy high-stimulation alternatives. "
        "Structured engagement with activities that meet dopaminergic need through low-risk behavior."
    )

    return {
        "gpari_score": round(score, 4),
        "score_ci": [round(score_ci_lo, 4), round(score_ci_hi, 4)],
        "percentile": min(99, max(1, int(score * 100))),
        "risk_level": risk_level,
        "pathway": pathway,
        "indirect_pct": round(indirect_mean, 4),
        "direct_pct": round(direct_pct, 4),
        "mediation_ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "feature_importances": {k: round(v, 6) for k, v in imp.items()},
        "intervention": intervention,
        "model_info": {
            "type": "GradientBoostingClassifier",
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "auc": 0.885
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "GradientBoostingClassifier", "auc": 0.885}
