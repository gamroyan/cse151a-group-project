## Milestone 3: Pre-Processing & First Model

**Preprocessing steps and model training is in the jupyter notebook linked in README.md**

**Question 3: Where does your model fit in the fitting graph? (Build at least one model with different hyperparameters and check for over/underfitting, pick the best model). What are the next models you are thinking of and why?**

- Answer: On the fitting curve, k = 2 shows overfitting, driving training error down but generalizes worse on the test set (MSE = 2.290, RMSE ≈ 1.513, R² = 0.771, ±1.0 = 79.0%). Increasing to k = 5 smooths the model and improves generalization (MSE = 1.974, RMSE ≈ 1.405, R² = 0.802, ±1.0 = 78.3%). Despite the slightly lower ±1.0 rate, k = 5 has clearly better aggregate error and explains more variance, so we select it. The next model we’re considering is a RandomForestRegressor on the same 80/20 split, should capture non-linear interactions across our log numerics and one-hots and reduce variance via bagging. We’ll compare its RMSE/R² to the k = 5 KNN and tune if it performs better.

**Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?**

- Answer: The first model is a decent baseline, but there’s room to improve it. We can try deeper trees and run a small grid search to tune depth, number of trees, and feature subsampling for the ideal hyperparameters for our data. We can process some of the features like recommendations/credits/keywords using encoding or TF-IDF to find the most significant words to improve our models since it may have a significant corellation. After cleaning NAs and clipping outliers, we could compare Random Forest, ExtraTrees, and a boosted model, keeping the best via cross-validated RMSE. These steps should realistically lower RMSE and bump R².
