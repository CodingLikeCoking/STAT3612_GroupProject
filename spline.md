# (MARS regression) Multivariate adaptive regression spline using pyearth
Contents:
1. data preprocessing functions
2. loading/extracting data (the cleaned data was not used in latter steps)
3. preprocessing the data using 4 methods:
    1. attribute reduction (n_components = 50) (variance explained > 0.9) (reduce the dimension into (#samples, 50, 1))
    2. time reduction (factor = 24) (reduce the dimension into (#samples, #attr, 1))
    3. attribute reduction, then time reduction
    4. time reduction, then attribute reduction
4. model training (using the 4 preprocessed datasets mentioned above)
5. full model training
6. R-squared score comparison:
    1. model 1: -0.42151839494688415
    2. model 2: -0.19997696342137927
    3. model 3: -0.16864823826425712
    4. model 4: 0.1543887423371778 (best model for the validation set)
    5. full model: 0.13712012727545042