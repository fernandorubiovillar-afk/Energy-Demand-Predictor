# Energy-Demand-Predictor
“Energy demand (MW) prediction from weekday, time and temperature using baseline and tree-based ML models.”
Estructura de carpetas recomendadas:
energy-demand-ml/<br>
├─ src/<br>
│  └─ 01_Energy_Demand.py<br>
├─ data/<br>
│  └─ demanda-maxima-de-mendoza-2022.csv   (opcional subirlo, ver nota)<br>
├─ images/<br>
│  └─ classification/   (se crea al ejecutar)<br>
├─ results/<br>
│  └─ series.txt        (se crea al ejecutar)<br>
├─ README.md<br>
├─ requirements.txt<br>
├─ .gitignore<br>

# Energy Demand Prediction (MW)

Small, reproducible ML pipeline to predict energy demand (MW) from:
- weekday (0=Mon..6=Sun)
- minute of day
- ambient temperature

Models trained and evaluated:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The script performs train/test split, cross-validation on train (RMSE) and final evaluation on test (RMSE and R²). It also generates and saves the main figures.

## Project structure
```bash
src/ -> Python script
data/ -> Input CSV (not included if restricted)
images/ -> Generated figures (created on run)
results/ -> Generated outputs (created on run)
```

## Setup

Create and activate a virtual environment (recommended), then install dependencies:


```bash
pip install -r requirements.txt
```
Place the dataset at:
```bash
data/demanda-maxima-de-mendoza-2022.csv
```
Then execute:
```bash
python src/01_Energy_Demand.py
```

Figures will be saved in:
```bash
images/classification/
```
## Dataset format

The raw CSV is expected to contain 4 columns:
- day
- hour
- temperature
- mw
