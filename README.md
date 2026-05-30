# Reading Motion in the Wild

**Activity Recognition from UCSD Phone and Watch Sensors**

**Analysis by Alpaslan Selvili Ozbas**

This project investigates whether smartphone and smartwatch telemetry can distinguish real-world human activity contexts. The analysis uses the UCSD ExtraSensory dataset, an in-the-wild mobile sensing dataset with minute-level examples from 60 users.

## Introduction

The UCSD ExtraSensory dataset contains phone and smartwatch sensor features collected in real-world conditions. Each row represents roughly a one-minute sensing window for one user, and each user has a separate compressed feature file. The dataset includes precomputed motion, location, audio, phone-state, and watch features, along with cleaned self-reported context labels.

The main question for this project is: **can passive phone and watch telemetry distinguish common activity contexts such as sitting, lying down, standing, walking, bicycling, and running?**

For the current analysis, I loaded all 60 user files from the primary ExtraSensory archive. The selected activity-label workflow starts with 377,346 rows. Of these, 306,594 rows have exactly one selected activity context and can be used for the multiclass prediction task.

## Data Cleaning and Exploratory Data Analysis

The target variable, `activity_context`, is engineered from six cleaned ExtraSensory label columns:

| Activity context | Activity type | Rows | Users | Share of modeling rows |
| --- | --- | ---: | ---: | ---: |
| Sitting | Stationary | 136,356 | 60 | 44.5% |
| Lying down | Stationary | 104,210 | 58 | 34.0% |
| Standing | Stationary | 37,782 | 60 | 12.3% |
| Walking | Active | 22,136 | 60 | 7.2% |
| Bicycling | Active | 5,020 | 25 | 1.6% |
| Running | Active | 1,090 | 26 | 0.4% |

Rows with none of the selected activity labels are excluded from the prediction dataset. In this selected label set, no rows had multiple selected activity labels, so no rows needed to be dropped for target ambiguity.

<iframe
  src="assets/plots/checkpoint_activity_context_distribution.html"
  width="100%"
  height="520"
  frameborder="0"
></iframe>

This distribution shows a major class imbalance: stationary activities, especially sitting and lying down, are much more common than bicycling and running. That imbalance affects both model evaluation and final-model design.

## Assessment of Missingness

This section will be completed for the final project. The missingness analysis will identify columns with non-trivial missingness, discuss whether at least one column may be NMAR, and use permutation tests to compare missingness patterns across other observed variables.

## Hypothesis Testing

The hypothesis test compares phone accelerometer magnitude during active contexts and stationary contexts.

**Null hypothesis:** Active and stationary contexts have similar mean phone accelerometer magnitude; any observed difference is due to random chance.

**Alternative hypothesis:** Active contexts have higher mean phone accelerometer magnitude than stationary contexts.

**Test statistic:** Mean `raw_acc:magnitude_stats:mean` for active rows minus mean `raw_acc:magnitude_stats:mean` for stationary rows.

I used a one-sided permutation test with 1,000 permutations and a significance level of 0.05. The test used 306,580 rows; 14 otherwise usable rows were excluded because they were missing the accelerometer feature.

| Quantity | Value |
| --- | ---: |
| Active rows | 28,245 |
| Stationary rows | 278,335 |
| Active mean accelerometer magnitude | 1.0419 |
| Stationary mean accelerometer magnitude | 0.9985 |
| Observed difference | 0.0434 |
| p-value | 0.0010 |

<iframe
  src="assets/plots/checkpoint2_accelerometer_permutation_test.html"
  width="100%"
  height="520"
  frameborder="0"
></iframe>

Since the p-value is below 0.05, we reject the null hypothesis. In this dataset, active contexts have a higher mean phone accelerometer magnitude than stationary contexts, and the observed difference is unlikely under random reassignment of active and stationary labels. This is evidence of an association, not proof of causation.

## Framing a Prediction Problem

The prediction problem is multiclass classification. The response variable is `activity_context`, with six possible classes: sitting, lying down, standing, walking, bicycling, and running.

At prediction time, the model should only use sensor and phone-state information from the one-minute window. Since `activity_context` is created from `label:*` columns, all `label:*` columns are excluded from the feature set to avoid leakage.

Because the classes are imbalanced, macro F1-score and balanced accuracy are better evaluation metrics than raw accuracy. These metrics give minority classes more influence when judging model performance.

## Baseline Model

The baseline model uses six original, leakage-safe features:

| Feature | Source family |
| --- | --- |
| `raw_acc:magnitude_stats:mean` | Phone accelerometer |
| `raw_acc:magnitude_stats:std` | Phone accelerometer |
| `proc_gyro:magnitude_stats:mean` | Phone gyroscope |
| `proc_gyro:magnitude_stats:std` | Phone gyroscope |
| `location:log_diameter` | Location |
| `discrete:app_state:is_active` | Phone state |

The baseline pipeline imputes missing numeric values with the median, standardizes the features, and fits a logistic regression classifier with class balancing. The model uses an 80/20 stratified train/test split.

| Metric | Value |
| --- | ---: |
| Train rows | 245,275 |
| Test rows | 61,319 |
| Macro F1-score | 0.292 |
| Balanced accuracy | 0.400 |

This baseline is useful but not strong enough to be the final model. The small feature set struggles to separate some stationary contexts, especially sitting and standing. For the final model, I plan to test richer motion, watch, location, audio, phone-state, and timestamp-derived features.

## Final Model

This section will be completed for the final project. The final model will add engineered features and tune model hyperparameters while using the same train/test split as the baseline. Candidate improvements include watch accelerometer features, motion frequency and entropy features, location movement features, phone-state indicators, audio features, and timestamp-derived features such as hour of day.

## Fairness Analysis

This section will be completed for the final project. The fairness analysis will compare model performance across two meaningful user or context groups using a permutation test and the final fitted model.
