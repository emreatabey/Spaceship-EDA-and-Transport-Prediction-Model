# Spaceship: EDA and Transport Prediction Model 🛸
This notebook focuses on exploratory data analysis (EDA), feature engineering, and classification modeling to predict which passengers were transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly.

## Workflow
* **Data Cleaning:** Handling missing values by leveraging group relationships (e.g., filling `HomePlanet` based on `Group`).
* **Feature Engineering:**
    * Created `Total_Expenses` by aggregating luxury amenities (RoomService, Spa, etc.).
    * Derived `GroupSize` and `IsAlone` from `PassengerId`.
    * Extracted `Deck`, `Num`, and `Side` from the `Cabin` feature.
    * Applied `AgeBand` binning to capture age-related transport trends.
* **Data Preprocessing:**
    * Label Encoding for categorical variables.
    * Boolean to integer conversion for binary features (`CryoSleep`, `VIP`).
    * Feature scaling and handling of skewed numerical distributions.
* **Model Training:** Comparison of multiple classification models:
    * Logistic Regression, Random Forest, SVM, KNN.
    * Advanced Gradient Boosting: **XGBoost, LightGBM, and CatBoost.**
* **Evaluation:** Model selection based on validation accuracy and final submission to the Kaggle leaderboard.

## Tech Stack
* **Language:** Python
* **Libraries:**
    * **Data Manipulation:** `Pandas`, `NumPy`
    * **Visualization:** `Matplotlib`, `Seaborn`
    * **Machine Learning:** `Scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`
  
## Dataset
* **Source:** [Kaggle: Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data)
  
## Evaluation Metric
* **Kaggle Public Score:** **0.80500** (Accuracy)

## Submission
The final predictions are generated and saved in a file named `submission.csv`. This file follows the exact format required by the Kaggle competition to be evaluated on the public leaderboard.

* **Format:** CSV (Comma-Separated Values)
* **Columns:**
    * `PassengerId`: Unique identifier for each passenger from the test set.
    * `Transported`: The model's binary prediction (`True` for transported, `False` for not transported).
