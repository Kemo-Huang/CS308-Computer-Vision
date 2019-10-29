import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################

    # features = 0
    # octave_layers = 3
    # contrast_threshold = 0.04
    # edge_threshold = 10
    # sigma = 1.6
    # sift_init_sigma = 0.5

    # # init Gaussian
    # sig_diff = np.sqrt(
    #     max(sigma * sigma - sift_init_sigma * sift_init_sigma * 4, 0.01))
    # resized = cv2.resize(
    #     image, (2 * image.shape[1], 2 * image.shape[0]), interpolation=cv2.INTER_LINEAR)
    # base = cv2.GaussianBlur(resized, 0, sig_diff)

    # # number of octaves
    # # for( size_t i = 0; i < keypoints.size(); i++ )
    # # {
    # #     KeyPoint& kpt = keypoints[i];
    # #     float scale = 1.f/(float)(1 << -firstOctave);
    # #     kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
    # #     kpt.pt *= scale;
    # #     kpt.size *= scale;
    # # }
    # for i in range(len(x)):
    #     scale = 0.5
    #     octave = 0

    # octaves = int(round(np.log(min(base.shape)) / np.log(2) - 2)) + 1

    # # build Gaussian pyramid
    # sig_len = octave_layers + 3
    # sig = [sigma] * sig_len
    # gpyr = [None] * octaves * sig_len
    # k = np.pow(2, 1 / octave_layers)
    # for i in range(1, octave_layers + 3):
    #     sig_prev = np.pow(k, i-1) * sigma
    #     sig_total = sig_prev * k
    #     sig[i] = np.sqrt(sig_total * sig_total - sig_prev * sig_prev)
    # for o in range(octaves):
    #     for i in range(sig_len):
    #         if o == 0 and i == 0:
    #             gpyr[o * sig_len + i] = base
    #         elif i == 0:
    #             src = gpyr[(o - 1) * sig_len + octave_layers]
    #             cv2.resize(src, (src.shape[1] // 2, src.shape[0] // 2),
    #                        gpyr[o * sig_len + i], interpolation=cv2.INTER_NEAREST)
    #         else:
    #             src = gpyr[o * sig_len + i - 1]
    #             gpyr[o * sig_len + i] = cv2.GaussianBlur(src, 0, sig[i])

    # # build DoG pyramid
    # dogpyr = [None] * octaves * (octave_layers + 2)
    # for a in range(len(dogpyr)):
    #     o = a // (octave_layers + 2)
    #     i = a % (octave_layers + 2)
    #     src1 = gpyr[o*(octave_layers + 3) + i]
    #     src2 = gpyr[o*(octave_layers + 3) + i + 1]
    #     cv2.subtract(src1, src2, dogpyr[o*(octave_layers + 2) + i])

    # SIFT descriptor
    n_angles = 8
    n_bins = 4
    n_samples = n_bins * n_bins
    alpha = 3
    n_pts = len(x)
    threshold = 0.2

    fv = np.zeros((n_pts, n_angles * n_samples))

    angles = np.linspace(0, 2*np.pi, n_angles)

    interval = np.linspace(2 / n_bins, 2, n_bins) - (1 / n_bins + 1)

    grid_x, grid_y = np.meshgrid(interval, interval)
    grid_x = grid_x.reshape((1, n_samples))
    grid_y = grid_y.reshape((1, n_samples))

    # histogram of oriented gradients
    ix = cv2.Sobel(image, -1, 1, 0)
    iy = cv2.Sobel(image, -1, 0, 1)
    magnitude = np.sqrt(np.square(ix) + np.square(iy))
    theta = np.arctan2(iy, ix)
    io = np.zeros((image.shape[0], image.shape[1], n_angles))

    for a in range(n_angles):
        tmp = np.power(np.cos(theta - angles[a]), alpha)
        tmp[tmp < 0] = 0
        io[:, :, a] = np.multiply(tmp, magnitude)

    for i in range(n_pts):
        # find coordinates of sample points
        grid_x_t = grid_x * feature_width + x[i]
        grid_y_t = grid_y * feature_width + y[i]
        # find coordinates of pixels
        x_l = int(max(np.floor(x[i] - feature_width - feature_width / 4), 0))
        x_h = int(min(np.ceil(x[i] + feature_width + feature_width / 4), image.shape[1]))
        y_l = int(max(np.floor(y[i] - feature_width - feature_width / 4), 0))
        y_h = int(min(np.ceil(y[i] + feature_width + feature_width / 4), image.shape[0]))
        grid_px, grid_py = np.meshgrid(
            np.linspace(x_l, x_h, x_h - x_l),
            np.linspace(y_l, y_h, y_h - y_l))
        n_pix = np.prod(grid_px.shape)
        grid_px = np.reshape(grid_px, (n_pix, 1))
        grid_py = np.reshape(grid_py, (n_pix, 1))
        # find distance
        dist_px = abs(np.tile(grid_px, (1, n_samples)) -
                      np.tile(grid_x_t, (n_pix, 1)))
        dist_py = abs(np.tile(grid_py, (1, n_samples)) -
                      np.tile(grid_y_t, (n_pix, 1)))
        # find weights
        weights_x = 1 - dist_px / feature_width / 2
        weights_x[weights_x < 0] = 0
        weights_y = 1 - dist_py / feature_width / 2
        weights_y[weights_y < 0] = 0
        weights = np.multiply(weights_x, weights_y)
        
        # multiply weights
        curr_sift = np.zeros((n_angles, n_samples))
        for a in range(n_angles):
            tmp = np.reshape(io[y_l: y_h, x_l: x_h, a], (n_pix, 1))
            tmp = np.tile(tmp, (1, n_samples))
            curr_sift[a, :] = np.sum(np.multiply(tmp, weights))
        fv[i, :] = np.reshape(curr_sift, (1, n_samples * n_angles))

    # normalize, threshold, normalize
    tmp = np.sqrt(np.sum(np.power(fv, 2), 1))
    fv_norm = np.divide(fv, np.tile(tmp, (fv.shape[1], 1)).T)
    fv_norm[fv_norm > threshold] = threshold
    tmp = np.sqrt(np.sum(np.power(fv_norm, 2), 1))
    fv = np.divide(fv_norm, np.tile(tmp, (fv.shape[1], 1)).T)

    return fv
