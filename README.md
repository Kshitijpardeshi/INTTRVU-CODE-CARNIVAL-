
# Smart Delivery Intelligence System

An AI-powered delivery time prediction and logistics optimization system built using XGBoost with advanced feature engineering, SHAP explainability, and an interactive Streamlit dashboard.

---

## Project Overview

The Smart Delivery Intelligence System predicts delivery time (ETA) and provides operational decision support through:

* ETA prediction
* Delay risk estimation
* Confidence interval analysis
* Live route visualization
* Driver comparison engine
* What-if scenario simulation
* AI-generated operational summary
* Downloadable PDF intelligence reports

This system transforms structured logistics data into predictive operational insights.

---

## Machine Learning Approach

### Algorithm

XGBoost Regressor

### Target Transformation

The target variable (Delivery Time) is log-transformed using `log1p()` to stabilize variance and reduce the impact of extreme values. Predictions are converted back to the original scale using `expm1()`.

This improves regression stability and performance on right-skewed delivery time distributions.

---

## Feature Engineering

The model includes advanced feature engineering techniques:

* Haversine distance calculation between restaurant and delivery location
* Driver efficiency score (Age × Rating)
* Distance squared term for non-linear behavior
* Interaction features:

  * Rating × Distance
  * Age × Distance
* Target encoding for city

These engineered features enhance the predictive capability of the model.

---

## Model Performance

* RMSE: 7.09 minutes
* R² Score: 0.39

Given the absence of real-time contextual features such as traffic, weather, and order preparation time, the model demonstrates stable predictive performance using structural logistics features.

---

## Model Explainability

SHAP (SHapley Additive exPlanations) was used to interpret feature importance and model behavior.

Key insights include:

* Distance is the strongest predictor of delivery time
* Higher driver ratings reduce expected delivery time
* Interaction features contribute to efficiency modeling

The system is transparent and interpretable rather than a black-box model.

---

## Interactive Dashboard Features

The application is built using Streamlit and includes:

* City-based delivery configuration
* Real-time ETA prediction
* Delay risk probability visualization
* Confidence interval estimation
* Live route map using Folium
* Driver comparison engine
* What-if simulation for distance variation
* AI-generated operational summary
* PDF report export

---

## Repository Structure

```
.
├── Dataset.csv
├── train.py
├── model.pkl
├── columns.pkl
├── app.py
├── eda.ipynb
├── Smart-Delivery-Intelligence-System.pptx
└── README.md
```

### File Descriptions

**train.py**
Contains the complete data cleaning, preprocessing, feature engineering, model training, evaluation, and model saving pipeline.

**eda.ipynb**
Contains exploratory data analysis and includes the same preprocessing, feature engineering, and modeling logic as `train.py`, along with visualizations and step-by-step analytical insights.

**app.py**
Streamlit-based interactive dashboard for delivery intelligence and decision support.

**model.pkl**
Serialized trained XGBoost model.

**columns.pkl**
Saved feature column structure used during training.

**Dataset.csv**
Dataset used for training and evaluation.

**Smart-Delivery-Intelligence-System.pptx**
Presentation slides used for hackathon demonstration.

---

## How to Run

### Install Dependencies

If using a requirements file:

```
pip install -r requirements.txt
```

Or manually:

```
pip install streamlit xgboost pandas numpy scikit-learn folium streamlit-folium reportlab joblib shap
```

---

### Train the Model (Optional)

```
python train.py
```

---

### Run the Application

```
streamlit run app.py
```

---

## Real-World Applications

* Food delivery platforms
* E-commerce logistics
* Fleet optimization systems
* Smart city logistics infrastructure

---

## Future Enhancements

* Real-time traffic integration
* Weather data incorporation
* Live GPS tracking
* Dynamic driver allocation algorithms
* Deployment as a scalable SaaS API

---

## Author

Kshitij Pardeshi,
Manmohan Bora
(MSc Data Science and Big Data Analytics)


