# Principal Component Analysis (PCA)

This project uses PCA in two experiments:

1. **Projecting correlated 2D data onto its principal axes**, to see what PCA does geometrically
2. **Reducing the 4-feature Iris dataset to 2 dimensions**, to use PCA for dimensionality reduction

Libraries: `numpy`, `matplotlib`, `scikit-learn` (`PCA`, `datasets`, `StandardScaler`).

---

## PCA in Short

PCA builds new features called principal components. They are linear combinations of the original features, orthogonal to each other, and ordered by how much variance they explain. The first component points in the direction where the data varies most, the second is perpendicular to it and captures the next most variation, and so on. There are at most `min(number of samples, number of features)` components.

With two features, `PC1 = a*X1 + b*X2`, and the weights `a`, `b` are in `pca.components_`.

The projection of a point `x` onto component `v_k` is `z_k = x . v_k`, and the projected point in the original space is `z_k v_k`.

**Explained variance ratio** of component k:

`lambda_k / (lambda_1 + ... + lambda_p)`

where `lambda_k` is the eigenvalue of the k-th component. Keeping the first m components loses `lambda_(m+1) + ... + lambda_p` of the variance, which is the smallest possible squared reconstruction error for an m-dimensional linear projection.

Note: in regression, explained variance is about the target `y`. In PCA there is no target; it is the variance of the input `X` captured by each component.

---

## Experiment 1: Correlated 2D Data

200 samples are drawn from a bivariate normal with mean `[0, 0]` and covariance `[[3, 2], [2, 2]]`. The features are correlated, so the cloud is elongated.

![Scatter Plot](scatter-plot.png)

`PCA(n_components=2)` gives:

- `PC1 = [0.7822, 0.6231]`, explained variance ratio **0.911**
- `PC2 = [-0.6231, 0.7822]`, explained variance ratio **0.089**

Since there are only 2 features, this is a rotation of the axes, not a reduction.

Each point is projected with dot products (`np.dot(X, components[0])`, `np.dot(X, components[1])`) and mapped back onto each component line:

`(x . PC1) PC1` and `(x . PC2) PC2`

![Correlated Projected Data](correlated-projected-data.png)

Red points are the projections onto PC1 and blue points are the projections onto PC2. Each projection is the closest point on that line. The red spread is much longer than the blue one, which matches PC1 explaining 91% of the variance.

---

## Experiment 2: Iris Dimensionality Reduction

The Iris dataset has 150 samples, 4 features, and 3 classes (setosa, versicolor, virginica).

Features are standardized with `StandardScaler` first, since PCA is based on variance and features with larger scales would otherwise dominate.

`PCA(n_components=2)` reduces each sample from 4D to 2D. The two components keep **95.81%** of the total variance.

![2D Reduction](2D-reduction.png)

Setosa is well separated. Versicolor and virginica are mostly separated with some overlap.

Running `PCA()` with all components gives the explained variance of each one:

![Explained Variance](explained-variance.png)

Blue bars are per-component ratios and the dashed red line is the cumulative ratio. PC1 and PC2 cover about 96%, PC3 adds a little, and PC4 almost nothing. So keeping 2 components is enough.

---

## Limitations of PCA

- Components are combinations of features, so they are harder to interpret.
- PCA is linear and can miss non-linear structure.
- PCA doesn't use labels, so it doesn't necessarily preserve class separation.

---

## Possible Improvements

- Compare with t-SNE or UMAP
- Reconstruct data from fewer components and measure the loss
- Test PCA without standardization
- Use PCA before a classifier and compare performance
