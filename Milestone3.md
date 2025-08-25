## Milestone 3: Pre-Processing & First Model

**Preprocessing steps and model training is in the jupyter notebook linked in README.md**

**Question 3: Where does your model fit in the fitting graph? (Build at least one model with different hyperparameters and check for over/underfitting, pick the best model). What are the next models you are thinking of and why?**

- Answer: On the fitting curve, our KNN with k = 2 sits on the overfitting side: it drives training error down but generalizes worse, since decisions hinge on very few neighbors and pick up noise. When we train k = 5 we move toward a better spot where we get MSE = 1.974 (RMSE ≈ 1.405), R² = 0.802, a better generalization than k = 2—so we select k = 5 as our current model. The next model we’re considering is a RandomForestRegressor on the same 80/20 split. A forest should capture non-linear interactions across our log numerics and one-hots, and reduce variance via bagging. We’ll compare its RMSE/R² to the k=5 KNN and, if it wins, tune the model further.
