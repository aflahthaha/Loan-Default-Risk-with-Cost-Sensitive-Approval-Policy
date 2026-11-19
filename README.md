# Loan Default Risk Prediction with a Cost-Sensitive Approval Policy

A machine-learning project that predicts whether a vehicle-loan borrower will default, and then turns that prediction into an **approval decision based on the bank's own costs**, instead of the usual fixed 0.5 cutoff. It ends with a Streamlit app where a bank enters its costs and the app finds the best cutoff.

**In one line:** the model ranks borrowers by risk (ROC-AUC about 0.65), and the cost-based cutoff makes the *decision* cheaper for the bank than the standard 0.5 rule.

---

## Why this project

Most loan-default projects stop at "accuracy" or "F1". A bank, however, does not care about those. It cares about money:

- Approving a borrower who defaults is **expensive** (the loan is lost).
- Rejecting a borrower who would have repaid is **cheaper** (lost profit), but it still costs something.

These two mistakes do not cost the same, so a cutoff of 0.5 is rarely the right one. This project picks the cutoff that gives the **lowest total cost** for the costs the bank states.

## Dataset

- **Source:** [L&T Vehicle Loan Default Prediction](https://www.kaggle.com/mamtadhaker/lt-vehicle-loan-default-prediction) (Kaggle; originally from the L&T Financial Services and Analytics Vidhya FinHack).
- **Size:** 233,154 loans, 41 columns.
- **Target:** `loan_default` (1 = the borrower defaulted on the first EMI). About **21.7%** of loans default, so the classes are imbalanced.
- **Not included in this repo.** Download the training file from Kaggle and save it next to the notebook as `loan_default_data.csv`.

## Approach

**1. Cleaning and feature engineering**
- Missing `Employment.Type` filled with `Unknown`.
- Two-digit birth years fixed (a date of birth in the future had 100 years subtracted), then `age` calculated at disbursal.
- Text durations such as `1yrs 11mon` converted to months.
- Dropped ID-like columns with thousands of unique values (`UniqueID`, `branch_id`, `supplier_id`, `Employee_code_ID`, `Current_pincode_ID`) and four flags (PAN, mobile number, passport, driving licence).
- Collapsed the secondary-account columns into one flag, `has_secondary_account`.
- Used the bureau score *description* (20 categories) instead of the raw bureau score.
- Removed one extreme outlier row.
- Dropped near-duplicate columns found with Spearman correlation (for example, `PRI.SANCTIONED.AMOUNT` against `PRI.DISBURSED.AMOUNT`, correlation 0.996).

**2. Two preprocessing pipelines (21 features, 72 columns after encoding)**

| Pipeline | Used for | Steps |
|---|---|---|
| A | Logistic regression | cap outliers at the 99.9th percentile, `log1p` for skewed columns, scale, one-hot encode |
| B | Random forest, XGBoost | one-hot encode only (trees need no scaling) |

All encoding and scaling sit inside scikit-learn `Pipeline` objects, so they are re-fitted inside each cross-validation fold and the validation data never leaks into training.

**3. Validation**
- Stratified 80/20 train-test split (seed 42): 186,523 training rows, 46,631 test rows.
- Cross-validation on the training set to compare models and tune them with `GridSearchCV`. The test set is used **once**, at the end.
- Models were compared on **ROC-AUC**, because it does not depend on a cutoff. Accuracy is not used: a model that says "nobody defaults" would be about 78% accurate and useless.

**4. Cost-based cutoff**
- Out-of-fold probabilities on the training set are used to try every cutoff from 0.05 to 0.95.
- For each cutoff: `total cost = missed defaulters x loss per default + wrongly rejected borrowers x lost gain`. The cheapest cutoff wins.

## Results

### Model comparison

The baseline is a dummy model that predicts the majority class. PR-AUC is scikit-learn's `average_precision`.

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test recall | Test precision |
|---|---|---|---|---|---|
| Dummy baseline | 0.500 | 0.500 | 0.217 | 0.000 | 0.000 |
| Logistic regression | 0.6409 | 0.6418 | 0.3148 | 0.641 | 0.290 |
| Random forest | 0.6464 | 0.6449 | 0.3245 | 0.606 | 0.295 |
| **XGBoost** | **0.6525** | **0.6511** | 0.3256 | 0.650 | 0.294 |
| Voting ensemble | 0.6523 | 0.6504 | 0.3264 | 0.643 | 0.295 |

Recall and precision above are at the default 0.5 cutoff. **XGBoost was chosen** (best CV ROC-AUC; the voting ensemble is no better and more complex). Best settings: `max_depth=3`, `learning_rate=0.1`, 300 trees, with class weighting for the imbalance.

### Cost-based cutoff (test set, 46,631 loans)

Costs are **assumptions**, not real bank data. A default costs the bank the stated loss and a wrongly rejected good customer costs ₹10,000.

| Loss per default | Best cutoff | Recall | Precision | Total cost at best cutoff | Total cost at 0.50 | Saving |
|---|---|---|---|---|---|---|
| ₹30,000 (3 : 1) | 0.54 | 0.525 | 0.315 | ₹259.52M | ₹264.60M | 1.9% |
| **₹50,000 (5 : 1)** | **0.41** | **0.840** | **0.263** | **₹319.42M** | ₹335.46M | 4.8% |
| ₹100,000 (10 : 1) | 0.27 | 0.977 | 0.228 | ₹357.82M | ₹512.61M | 30.2% |

At the main setting (5 : 1) the cutoff moves from 0.50 to 0.41. Missed defaulters drop from 3,543 to 1,620, and wrongly rejected borrowers rise from 15,831 to 23,842. **The more a default costs compared with a good customer, the lower the cutoff and the stricter the bank becomes.**

### A note on performance

Precision is low (about 0.26) and ROC-AUC is modest (about 0.65). That is expected here: the data only contains loans that were already **approved**, so many obviously risky applicants were filtered out before the data was collected, and there is no information from after the loan was given. The model still ranks borrowers better than chance (PR-AUC 0.33 against a 0.22 baseline), and the low precision at 0.41 is a deliberate trade for catching more defaulters, as the cost table shows.

### What drives the predictions

Top XGBoost feature importances (one-hot columns grouped back to the original feature):

| Feature | Importance |
|---|---|
| `State_ID` | 0.245 |
| `PERFORM_CNS.SCORE.DESCRIPTION` (bureau score category) | 0.228 |
| `manufacturer_id` | 0.108 |
| `Employment.Type` | 0.063 |
| `disbursed_amount` | 0.056 |
| `ltv` (loan-to-value) | 0.052 |

## SHAP analysis

SHAP analysis (2,000 test loans) ranks loan-to-value, State, and the bureau score category as the most influential features. A high LTV, overdue accounts, delinquent accounts, credit inquiries, a younger age and self-employment push the predicted risk up, while a longer credit history and “very low risk” bureau categories push it down. The strong effect of State_ID should be checked for fairness before real use.

## Streamlit app

The app in `deployment/` loads the saved model and lets a bank:
- enter its **loss per default** and **gain per good customer**, and get the best cutoff automatically;
- enter a borrower's details and see the **default probability**, an **approve / reject** decision and a risk band (High, Medium, Low);
- see **why** the model decided that way (which inputs raised or lowered the risk) in the extended version;
- see how the cost curve works and what the cutoff is worth for a number of applicants (an illustration based on the assumed costs, not a measured saving).

There are two versions: `app.py` (simple) and `app_wow.py` (extended, with tabs).

## Run it yourself

Python 3.10 or newer is needed.

```bash
git clone https://github.com/aflahthaha/Loan-Default-Risk-with-Cost-Sensitive-Approval-Policy.git
cd Loan-Default-Risk-with-Cost-Sensitive-Approval-Policy

# 1. Notebook: put the Kaggle training file here as loan_default_data.csv, then open
#    "Loan Default Risk with Cost-Sensitive Approval Policy.ipynb" and run all cells.

# 2. App
cd deployment
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

The saved model (`model.pkl`) was trained with **scikit-learn 1.7.2** and **NumPy below 2**. Keep those versions or the model will not load.

## Repository structure

```
.
├── Loan Default Risk with Cost-Sensitive Approval Policy.ipynb   # full analysis and modelling
├── deployment/
│   ├── app.py              # Streamlit app (simple)
│   ├── app_wow.py          # Streamlit app (extended)
│   ├── model.pkl           # trained pipeline (preprocessing + XGBoost)
│   ├── model_meta.json     # cutoff, default values, cost curve
│   └── requirements.txt
└── README.md
```

## Limitations

- **Approved loans only.** The model has never seen rejected applicants, so it cannot judge them well (selection bias).
- **Costs are assumed.** The cutoff is only as good as the cost numbers the bank provides.
- **Small leakage in preprocessing.** Outlier limits and the choice of columns to drop were worked out on the whole training set, so cross-validation folds see a little information from their validation part. The test set is untouched.
- **Probabilities are not calibrated.** Class weighting pushes the scores up, so the output is a risk *score*, not an exact probability of default. This is why the cutoff is found by scanning costs, not by a formula.
- **Fairness.** `State_ID` is the most important feature. A real lender would need to check whether using location produces unfair outcomes before using it.
- **Modest predictive power** (ROC-AUC about 0.65), as explained above.

## Future work

- Calibrate the probabilities and test on rejected-applicant data to reduce selection bias.
- Add a rule-based eligibility step before the model.
- Move outlier capping inside the pipeline and add a fairness check by region.
- Try more feature engineering and larger searches for gradient-boosting models.
- Deploy the app online.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, Altair, joblib.
