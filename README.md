# Agro AI — Crop Recommendation Tool

A single-page tool that recommends a crop for a field based on real soil and
climate readings. No backend, no database, no account — the trained machine
learning model runs entirely as JavaScript inside the page.

**Live demo:** open `agro-ai.html` directly in a browser, or host it as a
static file anywhere (GitHub Pages, Netlify, a plain S3 bucket — no build
step required).

## What it does

- Takes seven field readings — Nitrogen, Phosphorus, Potassium, temperature,
  humidity, soil pH, and rainfall — and returns a ranked crop recommendation
  with a confidence score.
- Shows *why* it made that call: which factors the model weighs most
  heavily overall, and whether the submitted readings fall inside or outside
  the typical range for the recommended crop.
- Offers indicative regional climate presets for people without a soil test
  kit, and a live weather lookup (via the free Open-Meteo API) that fills in
  real current temperature and humidity for the user's location.
- Works in English and Hindi.

## Why it's architected this way

This started as a Flask prototype with a login system, a SQLite user table,
and a "crop recommendation" button that was hardcoded to always return
"Wheat." The trained model existed in the repo but was never actually wired
into the app.

The rebuild removes all of that in favor of a zero-infrastructure design:

1. **No accounts, no server.** The tool has nothing worth logging into and
   nothing that needs saving between visits, so the entire login/session/
   database layer was cut rather than hardened.
2. **The model runs in the browser.** A scikit-learn `RandomForestClassifier`
   was trained on the dataset below, then compiled directly to a plain
   JavaScript function using [`m2cgen`](https://github.com/BayesWitnesses/m2cgen).
   There is no `/predict` API call — the compiled function runs client-side,
   so predictions are instant and nothing the user enters ever leaves their
   device.
3. **Everything real, nothing simulated.** The old market-price feature was
   fake hardcoded numbers and has been removed outright rather than kept as
   a placeholder. The weather feature was rewired to call Open-Meteo
   correctly (the old code pointed at a backend route that didn't exist).

## Model details

- **Algorithm:** RandomForestClassifier, 8 trees (reduced from a 100-tree
  version to keep the compiled JavaScript small — see trade-off note below).
- **Training data:** [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset),
  2,200 samples across 22 crops, with N, P, K, temperature, humidity, pH,
  and rainfall as features.
- **Accuracy:** 99.3% on a held-out 20% test split.
- **Trade-off:** dropping from 100 to 8 trees cost no measurable accuracy on
  this dataset (both scored 99.3% on the same test split) but reduced the
  compiled JS from ~2.2 MB to ~207 KB. If the model is ever retrained on a
  larger or messier dataset, re-check this trade-off — more classes or
  noisier data may need more trees, which means a larger embedded file.
- **Verification:** predictions from the compiled JavaScript were checked
  against the original Python model's `.predict()` output on a random
  sample and matched exactly, including predicted-class probabilities.

## Reproducing / retraining the model

```bash
pip install pandas scikit-learn m2cgen

python3 -c "
import pandas as pd, m2cgen as m2c
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('Crop_recommendation.csv')
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']

model = RandomForestClassifier(n_estimators=8, random_state=42)
model.fit(X, y)

with open('model.js', 'w') as f:
    f.write(m2c.export_to_javascript(model, function_name='score'))
"
```

The generated `score(input)` function takes a 7-element array in the order
`[N, P, K, temperature, humidity, ph, rainfall]` and returns an array of 22
scores in the class order scikit-learn assigns (`model.classes_`), which is
already embedded in `agro-ai.html`.

## Regional presets — an honest caveat

The "start from your region" presets fill in broad, indicative climate
figures (temperature, humidity, rainfall) for a dozen major Indian regions,
based on well-documented general climate patterns. They are a starting
point, not a substitute for a local forecast. Soil nutrient levels (N, P,
K) and pH are **not** estimated by region — they vary too much field to
field to guess responsibly. Use a soil test kit or a Soil Health Card value
for those.

## Tech stack

- Vanilla HTML/CSS/JS — no framework, no build tool, no dependencies to
  install to run it.
- Model training: Python, pandas, scikit-learn, m2cgen (only needed if
  retraining — not required to run the app).
- Weather: [Open-Meteo](https://open-meteo.com/) forecast API (no API key).

## File structure

```
agro-ai.html          the entire application
train_model.py        (optional) retrains the model from the CSV
Crop_recommendation.csv  training data
```

## License

MIT for the code in this repository. The dataset is subject to its own
license on Kaggle.
