# CS308-Computer-Vision

## Image Channels

RGB -> HSV (Hue, Saturation, Value)

- V = max
- S = max - min / max
- H = 
  - if max = R, 60 * (G - B) / (max - min)
  - if max = G, 120 * (B - R) / (max - min)
  - if max = B, 240 * (R - G) / (max - min)

## Histogram Equalization

**T(k) = floor((L - 1) sum(p0..k))**

- L = intensity
- pn = number of pixels with intensity n / total number of pixels

## Convolution

```python
h, w, channels = image.shape
fh, fw = kernel.shape[:2]
pad_h = (fh - 1) // 2
pad_w = (fw - 1) // 2
image = np.pad(image, [(pad_h, pad_h), (pad_w, pad_w), (0, 0)], 'symmetric')

kernel = kernel[..., None]  # [fH, fW] -> [fH, fW, 1]
output = np.zeros((h, w, channels), dtype='float32')
for r in range(h):
    for c in range(w):
        # image patch: [fH, fW, 3]
        # `filter`: [fH, fW, 1] -> [fH, fW, 3]
        result = image[r:r+fh, c:c+fw] * kernel
        output[r, c, :] = np.sum(np.sum(result, axis=0), axis=0)
```

## Filtering

**Gaussian filter**

- weights = 1/(2 * pi * std^2) * exp(-(x^2 + y^2) / (2 * std^2)) / max

**Sobel filter**

- mean = [1, 2, 1]
- gradient = [1 0 -1]
- horizontal sobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
- vertical sobel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

Laplacian of Gaussian (LoG) filter

- 2nd derivative = (x^2 + y^2 / std^4 - 2 / std^2) * G

Low pass

- Gaussian

High pass

- 1 - Gaussian

## Fourier

DFT

- X(k) = sum(n=0..N-1, x(n) * exp(-2pi * j * k * n / N)

IDFT

- x(n) = 1/N * sum(k=0..N-1, X(k) * exp(2pi * j * k * n / N) 

2D DFT

- F(u, v) = sum(x=0..M-1, y=0..N-1, f(x, y) * exp(-2pi * j * (ux / M + vy / N)))

Gabor

## Pyramid

Gaussian Pyramid

- scales (2^k)

- Gaussian filter with (2^k) stds

**Laplacian pyramid**

- interpolate

- DoG: G(k) - G(k-1)

Image Blending

- G(k) + L(k-1) + ... + L(1)

## Transformation

Affinity Transform

- I -> aI + b

Rotation Matrix

- R = [[cos, -sin], [sin, cos]]
- [x', y'] = R * [x, y]

Warping

## Keypoint Matching 

Interesting points

- repeatability
- distinctiveness

**Harris Corner Detection**

- Change in appearance of window w(x,y) for the shift [u,v]: 
  - E(u,v) = ∑ w(x, y)*[I(x+u, y+v) - I(x,y)]^2
  - w(x, y) = (1 or Gaussian) in window
- Second-order Taylor expansion of E(u,v) about (0,0):
  - E(u,v) = [u, v] M [u, v].T
  - M = ∑ w(x, y) [[Ix^2, Ix * Iy], [Ix * Iy, Iy^2]]
    - M =  [[grad(x)^2, grad(x) * grad(y)], [grad(x) * grad(y), grad(y)^2]]
    - Gaussian filtering
- Corner response function
  - R = det(M) − a * trace(M)^2
    - R = grad(x)^2 * grad(y)^2 - [grad(x) * grad(y)]^2 - a * [grad(x)^2 + grad(y)^2]^2
- R > threshold
- Take the points of local maxima (non-maximum suppression)

- Invariance:
  - invariant to translation and rotation
  - not invariant to scaling

**SIFT Descriptor**

- gradient orientation histogram
  - 4x4x8=128 array weighted by gradient magnitude
- define feature width for each keypoint
- normalize, threshold, normalize

**Matching**

- Nearest Neighbor Distance Ratio
  - NN1 / NN2

## RANSAC



## Hough Transform

shape detection

voting scheme

p = xcos(theta) + ysin(theta) 

## Manifold Learning

Dimensionality Reduction

Unsupervised learning + continuous

Linear methods

- **Principal component analysis (PCA)**
  - find the principal axes are those orthonormal axes onto which the variance retained under  projection is maximal
  - cov(X, Y) = 1/n * ∑ (Xi - Xm) * (Yi - Ym)
- Multidimensional scaling (MDS)

Nonlinear methods

- Kernel PCA
- Locally linear embedding (LLE)
- Isomap
- Laplacian eigenmaps (LE)
- **T-distributed stochastic neighbor embedding (TSNE)**

## Classification

Supervised learning + discrete

**Support Vector Machine**

- 2 classes

- for support vector x, wx + b = +-1 -> w (x+ - x-) = 2 -> Margin = 2 / |w|

- min{1/2 * |w|^2}, y(wx+b) >=1

- Lagrange: a>=0, L = 1/2 * |w|^2 - ∑ {a*(y-wx+b) - 1)}

  - dual problem: min(w,b) max(a>=0) L -> max(a>=0) min(w,b) L
  - derivatives = 0 -> w = ∑ a * y * x, ∑ a * y = 0
  - L = max(a>=0) ∑ a - 1/2 * ∑∑ {ai * aj * yi * yj * xi * xj}
  - KKT -> a(y(wx+b) - 1) = 0 (complementary slackness)
  - f = ∑ a * yi * xi * x + b
  - SMO algorithm

- Soft margin

  - slack variables ξ

  - min{1/2 * |w|^2 + C * ∑ξ}, y(wx+b) >= 1 - ξ, ξ >=0
  - 0 <= a <= C

- Non-linear classification

  - K(x, z) = Φ(x) * Φ(z), x = input space, z = feature space, Φ = feature map
  - K is semi-positive definite symmetric function
  - L = max(a>=0) ∑ a - 1/2 * ∑∑ {ai * aj * yi * yj * K(xi, xj)}
  - f = ∑ a * yi * K(xi, x) + b

- Multi-class classification

  - m class -> m SVMs

## Clustering

unsupervised learning + discrete

- Applications
  - summary
  - counting
  - segmentation
  - prediction
- K-means
- Mean-shift