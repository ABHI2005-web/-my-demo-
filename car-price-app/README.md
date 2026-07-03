# Car Price Predictor

A complete Flask web app that predicts a used car's resale price using a
RandomForest regression model, ready to deploy on Render.

## Project structure

```
car-price-app/
├── app.py                # Flask app (serves the site + prediction route)
├── train_model.py        # Generates dataset + trains + saves the model
├── requirements.txt
├── Procfile               # For Render/Heroku-style start command
├── render.yaml            # Render deployment config (auto build+start)
├── model/                 # Created after training (car_price_model.pkl, encoders.pkl)
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Run locally

```bash
pip install -r requirements.txt
python train_model.py      # trains the model, creates model/car_price_model.pkl
python app.py               # starts the site at http://localhost:5000
```

## Deploy on Render

1. Push this folder to a GitHub repository.
2. On Render.com, click **New +** → **Web Service**, connect your repo.
3. Render will auto-detect `render.yaml`. If not, set manually:
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `gunicorn app:app`
   - **Environment:** Python 3
4. Click **Create Web Service**. Once deployed, Render gives you a public
   URL like `https://car-price-predictor.onrender.com` — that's your
   submission link.

## Notes

- The model is trained on a synthetic but realistic dataset (price
  depreciation by age, mileage, fuel type, transmission, ownership, etc.)
  generated in `train_model.py`. If you have a real dataset (e.g. the
  CarDekho/Kaggle car dataset) with columns `Car_Name, Year, Selling_Price,
  Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner`,
  drop it in as `data.csv` and swap the synthetic-generation block in
  `train_model.py` for `pd.read_csv("data.csv")` — the rest of the
  pipeline works unchanged.
- Model files are generated at build time on Render (not committed), so
  the repo stays lightweight.
