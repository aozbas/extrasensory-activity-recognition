# Reading Motion in the Wild

**Activity Recognition from UCSD Phone and Watch Sensors**

**Analysis by Alpaslan Selvili Ozbas**

This project investigates whether smartphone and smartwatch telemetry can distinguish real-world human activity contexts. The analysis uses the UCSD ExtraSensory dataset, an in-the-wild mobile sensing dataset with minute-level examples from 60 users.

## Introduction

The UCSD ExtraSensory dataset contains phone and smartwatch sensor features collected in real-world conditions. Each row represents roughly a one-minute sensing window for one user, and each user has a separate compressed feature file. The dataset includes precomputed motion, location, audio, phone-state, and watch features, along with cleaned self-reported context labels.

The main question for this project is: **can passive phone and watch telemetry distinguish common activity contexts such as sitting, lying down, standing, walking, bicycling, and running?**

For the current analysis, I loaded all 60 user files from the primary ExtraSensory archive. The selected activity-label workflow starts with 377,346 rows. Of these, 306,594 rows have exactly one selected activity context and can be used for the multiclass prediction task.

## Data Cleaning and Exploratory Data Analysis

I loaded the 60 compressed per-user `.csv.gz` files from the primary ExtraSensory archive and added a `uuid` column from each filename. The cleaning workflow keeps the selected activity-label columns, parses the Unix `timestamp` column into a datetime, and creates a single engineered response column named `activity_context`. Rows with no selected activity label are excluded from the prediction dataset. In this selected label set, no rows had multiple selected activity labels, so no rows needed to be dropped for target ambiguity.

The selected numeric sensor fields already use standard missing values, so I kept missing sensor readings as `NaN` rather than replacing them with artificial constants. The `label:*` columns are used only to construct the response variable and are excluded from the model feature set to avoid leakage.

A small sample of the cleaned modeling table is shown below. The user identifier is shortened for readability.

| user | timestamp | activity_context | activity_type | phone_accel_mean | phone_gyro_std | location_log_diameter | phone_app_active |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 00EABED2 | 2015-10-05 21:06:01 | Sitting | Stationary | 0.997 | 0.002 | 2.313 | 0 |
| 00EABED2 | 2015-10-05 21:07:01 | Sitting | Stationary | 0.997 | 0.001 | 2.263 | 1 |
| 00EABED2 | 2015-10-05 21:08:01 | Sitting | Stationary | 0.997 | 0.002 | -0.565 | 1 |
| 00EABED2 | 2015-10-05 21:09:01 | Sitting | Stationary | 0.997 | 0.002 | 0.741 | 1 |
| 00EABED2 | 2015-10-05 21:10:31 | Sitting | Stationary | 0.997 | 0.341 | 1.612 | 0 |

The target variable, `activity_context`, is engineered from six cleaned ExtraSensory label columns:

| Activity context | Activity type | Rows | Users | Share of modeling rows |
| --- | --- | ---: | ---: | ---: |
| Sitting | Stationary | 136,356 | 60 | 44.5% |
| Lying down | Stationary | 104,210 | 58 | 34.0% |
| Standing | Stationary | 37,782 | 60 | 12.3% |
| Walking | Active | 22,136 | 60 | 7.2% |
| Bicycling | Active | 5,020 | 25 | 1.6% |
| Running | Active | 1,090 | 26 | 0.4% |

<iframe
  src="assets/plots/checkpoint_activity_context_distribution.html"
  width="100%"
  height="520"
  frameborder="0"
></iframe>

This distribution shows a major class imbalance: stationary activities, especially sitting and lying down, are much more common than bicycling and running. That imbalance affects both model evaluation and final-model design.

The timestamp-derived hour distribution shows that the modeling rows are spread throughout the day, with the largest hourly count at 9 PM and the smallest at 2 PM.

<iframe
  src="assets/plots/eda_hour_of_day_distribution.html"
  width="100%"
  height="520"
  frameborder="0"
></iframe>

The grouped table below summarizes several sensor features by activity context. Active contexts have larger average phone accelerometer and gyroscope variation, which motivates using richer motion features in the final model. Location missingness also varies by activity context, which is examined more formally in the missingness section.

| Activity context | Rows | Users | Mean phone accel | Mean phone gyro std | App active rate | Location missing rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sitting | 136,356 | 60 | 0.998 | 0.136 | 2.7% | 10.6% |
| Lying down | 104,210 | 58 | 0.997 | 0.041 | 0.6% | 10.5% |
| Standing | 37,782 | 60 | 1.005 | 0.267 | 2.6% | 13.5% |
| Walking | 22,136 | 60 | 1.040 | 0.605 | 2.8% | 16.6% |
| Bicycling | 5,020 | 25 | 1.032 | 0.400 | 1.1% | 5.1% |
| Running | 1,090 | 26 | 1.129 | 0.658 | 1.1% | 7.9% |

<iframe
  src="assets/plots/eda_accelerometer_by_activity.html"
  width="100%"
  height="520"
  frameborder="0"
></iframe>

## Assessment of Missingness

I focused the missingness analysis on `location:log_diameter`, the location feature used in the baseline model. This column is missing in 34,434 of the 306,594 modeling rows, or 11.2% of the activity-labeled rows. Because this feature summarizes the spatial spread of location updates within a sensing window, missing values may occur when location services are unavailable, disabled, inaccurate, or not collected during that minute.

At least some location-feature missingness may be `NMAR`: whether location is missing can plausibly depend on the unobserved true location context or user behavior, such as being indoors, choosing not to share location, or moving through places where GPS is unreliable. The permutation tests below do not prove or disprove NMAR; they test whether the missingness indicator is associated with other observed variables.

| Activity context | Activity type | Rows | Missing rows | Missing rate |
| --- | --- | ---: | ---: | ---: |
| Walking | Active | 22,136 | 3,671 | 16.6% |
| Standing | Stationary | 37,782 | 5,086 | 13.5% |
| Sitting | Stationary | 136,356 | 14,419 | 10.6% |
| Lying down | Stationary | 104,210 | 10,917 | 10.5% |
| Running | Active | 1,090 | 86 | 7.9% |
| Bicycling | Active | 5,020 | 255 | 5.1% |

<iframe
  src="assets/plots/missingness_location_log_diameter_by_activity.html"
  width="100%"
  height="520"
  frameborder="0"
></iframe>

I ran two-sided permutation tests using the difference in missing rates as the test statistic.

| Test | Group comparison | Observed difference | p-value | Conclusion |
| --- | --- | ---: | ---: | --- |
| Dependency test | Active rows minus stationary rows | 0.0327 | 0.0010 | Location missingness differs by activity type. |
| Non-dependency test | 6 AM-noon rows minus all other rows | -0.0005 | 0.7393 | No meaningful missingness difference was detected for this time-of-day split. |

The first test suggests that `location:log_diameter` missingness depends on observed activity type: active rows have a higher missing rate than stationary rows. The second test gives a useful contrast: the same missingness indicator does not appear to depend on whether the row was recorded between 6 AM and noon.

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

All 6 baseline features are quantitative. The baseline uses 0 ordinal features and 0 nominal features, so no categorical encoding is needed.

| Metric | Value |
| --- | ---: |
| Train rows | 245,275 |
| Test rows | 61,319 |
| Macro F1-score | 0.292 |
| Balanced accuracy | 0.400 |

This baseline is useful but not strong enough to be the final model. The small feature set struggles to separate some stationary contexts, especially sitting and standing. The final model below expands the feature set while keeping the same held-out test rows, leakage controls, and evaluation metrics.

## Final Model

The final model uses the same 80/20 stratified train/test split as the baseline, reusing the exact baseline train and test row indices. It also keeps all `label:*` columns excluded from the feature set.

I used one `sklearn` pipeline with three stages: engineered feature creation, median imputation, and an extremely randomized trees classifier. This model is a better fit for the activity-recognition task than a purely linear model because it can learn nonlinear thresholds and interactions among motion, watch, location, time, and phone-state signals. ExtraTrees also averages many randomized decision trees, which helps reduce overfitting on the many correlated sensor features. The engineered features are:

| Engineered feature | Description |
| --- | --- |
| `has_watch_acceleration` | Whether watch accelerometer features were observed for the row. |
| `has_location_signal` | Whether `location:log_diameter` was observed for the row. |
| `phone_motion_intensity` | Row-wise sum of phone accelerometer and gyroscope standard deviation. |

The final model includes 123 total features: 120 original ExtraSensory columns plus the 3 engineered features.

| Source family | Feature count |
| --- | ---: |
| Watch accelerometer | 46 |
| Phone accelerometer | 26 |
| Phone gyroscope | 26 |
| Phone state | 11 |
| Time of day | 8 |
| Location | 3 |
| Engineered | 3 |

I tuned the ExtraTrees model with `GridSearchCV`, using 3-fold stratified cross-validation on the training split and macro F1-score as the selection metric. The planned grid was `n_estimators = [200, 400]`, `max_depth = [30, None]`, `min_samples_leaf = [2, 3, 5]`, `max_features = ["sqrt", 0.5]`, and `class_weight = ["balanced_subsample"]`. These control the number of trees, tree depth, leaf size, feature subsampling, and class-imbalance handling. The best hyperparameters were:

| Hyperparameter | Selected value |
| --- | --- |
| `n_estimators` | `400` |
| `max_depth` | `30` |
| `max_features` | `0.5` |
| `min_samples_leaf` | `2` |
| `class_weight` | `balanced_subsample` |

| Metric | Baseline | Final model | Improvement |
| --- | ---: | ---: | ---: |
| Macro F1-score | 0.292 | 0.812 | +0.521 |
| Balanced accuracy | 0.400 | 0.774 | +0.373 |

<iframe
  src="assets/plots/final_model_baseline_comparison.html"
  width="100%"
  height="520"
  frameborder="0"
></iframe>

The final model is a clear improvement over the baseline. The richer motion, watch, time, phone-state, and location features give the model more information about each one-minute window, while the engineered availability features make missing sensor signals explicit. The ExtraTrees model also captures nonlinear combinations of those signals, such as large phone motion together with observed watch acceleration. The final model improves every activity class compared with the earlier baseline and remains reproducible and leakage-safe.

## Fairness Analysis

This section will be completed for the final project. The fairness analysis will compare model performance across two meaningful user or context groups using a permutation test and the final fitted model.
