# ICU Nutritional Risk & Caloric Delivery: Mortality Prediction Model

## Overview
This repository contains the Python-based statistical methodology and data analysis pipeline used for predicting 28-day mortality in critically ill Intensive Care Unit (ICU) patients. The study investigates the independent predictive power of conventional nutritional risk scores (mNUTRIC, NRS-2002, MNA) versus actual caloric delivery ratios, while strictly controlling for clinical severity (SOFA, APACHE-II) and sepsis.

## Methodological Highlights
The codebase is designed to rigorously address common statistical pitfalls in clinical and epidemiological research:

1. **Prevention of Multicollinearity:** Nutritional risk scores that intrinsically incorporate clinical severity or age (e.g., the SOFA component within the mNUTRIC score) were deliberately excluded from the core multivariate regression model. This isolates the true independent prognostic value of the *Caloric Delivery Ratio*.
2. **Selection Bias & Missing Data Analysis:** The script includes an internal data audit to evaluate the missing data mechanism. It statistically compares the excluded cohort (due to missing caloric records) against the included complete-case cohort. The analysis demonstrates that the excluded patients represented a distinct sub-population with significantly lower clinical severity (SOFA/APACHE-II), thereby paradoxically strengthening the clinical relevance of caloric delivery in the remaining severe cohort.
3. **Age-Stratified Subgroup Analysis:** Includes a dedicated geriatric (≥65 years) subgroup analysis, revealing the shifting prognostic weight from clinical severity (SOFA) to baseline nutritional status (MNA).

## Requirements
The script requires Python 3.x and the following libraries:
* `pandas`
* `numpy`
* `scipy`
* `scikit-learn`
* `statsmodels`

## Usage
The main script is designed for cross-platform compatibility, including local environments and Google Colab.

1. Clone the repository or run the script directly in a Jupyter Notebook / Google Colab.
2. Execute the script.
3. When prompted, upload or specify the path to the clinical dataset (Excel/CSV). 
   * *Note: The dataset must contain relevant columns such as Age, SOFA, APACHE-II, mNUTRIC, NRS-2002, MNA, Caloric Targets, and 28-Day Mortality.*

## Output Structure
Upon execution, the script systematically generates the following analytical sections:
* **Section 1:** Selection Bias Control (Mann-Whitney U tests for missing data).
* **Section 2:** Univariate Analysis (Medians and P-values for all continuous clinical variables).
* **Section 3:** Multivariate Logistic Regression (Odds Ratios, 95% CIs, and P-values for the core predictive model).
* **Section 4:** ROC Curve Analysis (AUC values for scoring systems across general and geriatric populations).

## Ethical Disclaimer
🚨 **Patient Privacy:** This repository strictly contains the analytical code and methodology. Due to medical ethics and patient confidentiality regulations, **the actual clinical dataset containing protected health information (PHI) is NOT included and will not be shared publicly.** Researchers wishing to test the code should structure their own mock datasets matching the variables described in the methodology.
