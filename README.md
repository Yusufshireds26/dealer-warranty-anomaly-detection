# Dealer Warranty Anomaly Detection Framework

## Overview

This project developed a dealer-level anomaly detection framework using warranty claims data to identify dealers exhibiting unusual warranty behavior.

The goal was to support proactive warranty monitoring by identifying dealers with abnormal cost, reliability, and claims-processing patterns using machine learning.

---

## Business Problem

How can we identify dealers with unusual warranty behavior and prioritize them for investigation?

This project analyzed warranty claims and dealer performance metrics to create a data-driven dealer watchlist and risk scoring framework.

---

## Dataset

- 105,000+ Warranty Claims
- 74 Dealers
- Q1-Q2 2026 Analysis
- Consolidated Claims Data

---

## Features Engineered

### Warranty Cost Metrics
- CostPerVIN
- AvgCostPerClaim
- ClaimsPerVIN

### Reliability Metrics
- AvgMilesToFailure
- AvgDaysToFailure
- RepeatClaimPct
- RepeatVINPct

### Claims Processing Metrics
- FailuretoAcceptedDays
- AccepttoApprovedDays
- AvgClaimTouches

### Warranty Complexity Metrics
- ContractCodeCount
- ClaimsPerContractCode
- UnknownPct

---

## Methodology

The project followed the CRISP-DM framework:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

---

## Models Used

### Isolation Forest
Identified dealers that were significantly different from the overall dealer population.

### Local Outlier Factor (LOF)
Identified dealers with unusual behavior relative to similar dealers.

### DBSCAN
Identified dealers that did not fit into normal dealer clusters.

### One-Class SVM
Identified dealers that fell outside the learned normal dealer profile.

---

## Consensus Framework

A consensus scoring methodology was developed by combining outputs from all four models.

Higher consensus scores indicate stronger evidence that a dealer represents a true anomaly.

---

## Key Findings

Top anomaly drivers included:

1. AvgMilesToFailure
2. CostPerVIN
3. AvgCostPerClaim
4. FailuretoAcceptedDays
5. AccepttoApprovedDays

The newly engineered claims-processing metrics emerged as significant drivers of anomalous dealer behavior.

---

## Methodology (CRISP-DM)

### Business Understanding
Define anomaly detection objectives.

### Data Understanding
Explore data and identify patterns.

### Data Preparation
Clean data and engineer features.

### Modeling
Isolation Forest, LOF, DBSCAN, One-Class SVM.

### Evaluation
Feature importance and consensus scoring.

### Deployment
Watchlists, rankings, and reporting.

## Deliverables

- Dealer Watchlist
- Consensus Scorecard
- Top Dealer Deep Dive Analysis
- Cross-Model Feature Importance Analysis
- Executive Presentation

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- OpenPyXL
- SQL Server
- Excel

---

## Skills Demonstrated

- Machine Learning
- Anomaly Detection
- Feature Engineering
- Data Preparation
- Data Visualization
- Business Analytics
- Model Evaluation
- Executive Reporting
