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
    ksize = 3
    n_angles = 8
    n_bins = 4
    n_samples = n_bins * n_bins
    n_pts = len(x)
    threshold = 0.2

    fv = np.zeros((n_pts, n_angles * n_samples))

    # padding
    image = np.pad(image, feature_width // 2)

    # histogram of oriented gradients
    ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(np.square(ix) + np.square(iy))
    theta = np.arctan2(iy, ix)

    theta[theta > 1] = 2
    theta[theta < -1] = -1
    x = x.astype(int)
    y = y.astype(int)

    for k in range(n_pts):
        histogram = np.zeros((n_bins, n_bins, n_angles))
        for j in range(feature_width):
            for i in range(feature_width):
                curr = (y[k] + j, x[k] + i)
                if ix[curr] > 0:
                    histogram[j // n_bins, i // n_bins,
                              int(np.ceil(theta[curr] + 1))] += magnitude[curr]
                else:
                    histogram[j // n_bins, i // n_bins,
                              int(np.ceil(theta[curr] + 1 + n_bins))] += magnitude[curr]
        fv[k, :] = np.reshape(histogram, (1, n_angles * n_samples))

    # normalize, threshold, normalize
    tmp = np.sqrt(np.sum(np.power(fv, 2), 1))
    fv_norm = np.divide(fv, np.tile(tmp, (fv.shape[1], 1)).T)
    fv_norm[fv_norm > threshold] = threshold
    tmp = np.sqrt(np.sum(np.power(fv_norm, 2), 1))
    fv = np.divide(fv_norm, np.tile(tmp, (fv.shape[1], 1)).T)
    return fv
