import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    
    # dest = cv2.cornerHarris(image, 5, 3, 0.06) 
    # ind = np.argwhere(dest > 0.01 * dest.max())
    # x = ind[:, 0]
    # y = ind[:, 1]
    
    k = 0.04
    thres_w = 0.01
    sigma = 1.5
    nms_w = 0.4

    # Compute the horizontal and vertical derivatives of the image I x and I y
    # by convolving the original image with derivatives of Gaussians
    ix = cv2.Sobel(image, -1, 1, 0)
    iy = cv2.Sobel(image, -1, 0, 1)

    # Compute the three images corresponding to the outer products of these gradients.
    # (The matrix A is symmetric, so only three entries are needed.)
    ixx = np.square(ix)
    iyy = np.square(iy)
    ixy = np.multiply(ix, iy)

    # Convolve each of these images with a larger Gaussian.
    gaussian = cv2.getGaussianKernel(3, sigma)
    gxx = cv2.filter2D(ixx, -1, gaussian)
    gyy = cv2.filter2D(iyy, -1, gaussian)
    gxy = cv2.filter2D(ixy, -1, gaussian)

    # Compute a scalar interest measure using one of the formulas discussed above.
    # np.linalg.det(A) - k * (np.trace(A) ** 2)
    R = np.multiply(gxx, gyy) - np.square(gxy) - k * np.square(gxx + gyy)

    # Find local maxima above a certain threshold and report them as detected feature point locations.
    thres = thres_w * R.max()
    indices = np.argwhere(R > thres)
    x = indices[:, 1]
    y = indices[:, 0]

    responses = R[y, x]

    # non-maxima suppress
    size = len(indices)
    ind = np.argsort(-responses)
    points = np.hstack((y[ind], x[ind]))
    
    radii = np.zeros(size)
    radii[0] = np.inf

    for i in range(1, size):
        curr_response = responses[ind[i]]
        idx = i
        while idx < size - 1:
            if responses[idx+1] * 1.1 > curr_response:
                idx += 1
            else:
                break
        radii[i] = np.min(np.square(points[:idx] - points[i]))

    n = int(nms_w * size)
    x = np.array(x[np.argpartition(radii, -n)[-n:]])
    y = np.array(y[np.argpartition(radii, -n)[-n:]])

    return x, y, confidences, scales, orientations
