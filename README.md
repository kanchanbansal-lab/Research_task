# Multicopter Performance Modeling and Prediction

## Project Overview

This project focuses on the development and validation of **multicopter performance prediction models** using both **physics-based** and **data-driven** approaches. The goal is to accurately model flight energy consumption.

The Dataset of Rodrigues et. al. 2021 [1] serves as the primary reference platform for model development.

[1] Rodrigues, T.A., Patrikar, J., Choudhry, A. et al. In-flight positional and energy use data set of a DJI Matrice 100 quadcopter for small package delivery. Sci Data 8, 155 (2021). https://doi.org/10.1038/s41597-021-00930-x

---

## Objectives

* Develop a **physics-based flight performance model** for a multicopter UAV
* Build **data-driven machine learning models** using real flight data
* Compare predictive performance between physics-based and data-driven approaches

---

## Physical Modeling Approach

### Parameters

Model parameters are obtained from:

* Manufacturer technical manuals
* Published research papers

Key parameters include:

* Vehicle mass
* Rotor and propeller characteristics
* Thrust and power coefficients
* Battery specifications

---

## Data-Driven Modeling Approach

### Machine Learning Models

The following algorithms are implemented and trained:

* **Random Forest**

These models learn the relationship between flight conditions and performance metrics directly from data.

---

## Repository Structure (Proposed)

```
├── data/               # Raw and processed flight data
├── old_notebooks/      # old notebooks
├── papers/             # literature used in this project
├── presentation/       # final presentation
├── report/             # final report
├── README.md
└── data_analysis_report.ipynb
```