# Concert Revenue Prediction

This is my first machine learning project.

The goal of this project is to clean and analyze concert tour data, then build a simple linear regression model to estimate gross revenue based on the number of shows.

## What I did

* Loaded and cleaned raw concert tour data
* Removed symbols, commas, and reference marks from numeric columns
* Converted important columns to numeric format
* Cleaned artist names, tour titles, rankings, and year values
* Analyzed artist performance using grouping and aggregation
* Visualized total gross revenue by artist
* Built a simple machine learning model using linear regression

## Machine Learning

The model uses:

* **Feature (`X`)**: number of shows
* **Target (`y`)**: actual gross revenue

The dataset was split into training and testing sets.
A linear regression model was trained to estimate revenue from the number of shows.

## Tools Used

* Python
* pandas
* matplotlib
* scikit-learn

## Project Files

* `main.py` — main analysis and machine learning code
* `cleaned_concert_data.csv` — cleaned dataset
* `concert_data.csv` — raw dataset

## Notes

This is my first machine learning project, so the main goal was learning the workflow of:

data cleaning → exploration → visualization → modeling

The dataset is small, so predictions should be treated as learning results rather than production-level forecasts.
