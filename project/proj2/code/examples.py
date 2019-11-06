import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import *
from student_feature_matching import match_features
from student_sift import get_features
from student_harris import get_interest_points


scale_factor = 0.5
feature_width = 16  # width and height of each local feature, in pixels.

images = [
    ['../data/Mount Rushmore/9021235130_7c2acd9554_o.jpg',
        '../data/Mount Rushmore/9318872612_a255c874fb_o.jpg'],
    ['../data/Episcopal Gaudi/4386465943_8cf9776378_o.jpg',
        '../data/Episcopal Gaudi/3743214471_1b5bbfda98_o.jpg'],
    ['../data/Capricho Gaudi/36185369_1dcbb23308_o.jpg',
        '../data/Capricho Gaudi/6727732233_4564516d61_o.jpg']
]

for i, pair in enumerate(images):
    image1 = load_image(pair[0])
    image2 = load_image(pair[1])

    print(f"\nstart matching image pair {i}")
    start_time = time.time()

    image1 = cv2.resize(image1, (0, 0), fx=scale_factor, fy=scale_factor)
    image2 = cv2.resize(image2, (0, 0), fx=scale_factor, fy=scale_factor)
    image1_bw = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2_bw = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    x1, y1, _, _, _ = get_interest_points(image1_bw, feature_width)
    x2, y2, _, _, _ = get_interest_points(image2_bw, feature_width)

    print('{:d} corners in image 1, {:d} corners in image 2'.format(len(x1), len(x2)))

    image1_features = get_features(image1_bw, x1, y1, feature_width)
    image2_features = get_features(image2_bw, x2, y2, feature_width)

    matches, _ = match_features(
        image1_features, image2_features, x1, y1, x2, y2)
    print('{:d} matches from {:d} corners'.format(len(matches), len(x1)))

    print(f"time cost: {time.time() - start_time}")

    num_pts_to_visualize = 100
    c1 = show_correspondence_circles(image1, image2,
                                     x1[matches[:num_pts_to_visualize, 0]],
                                     y1[matches[:num_pts_to_visualize, 0]],
                                     x2[matches[:num_pts_to_visualize, 1]],
                                     y2[matches[:num_pts_to_visualize, 1]])
    plt.figure()
    plt.imshow(c1)
    plt.savefig(f'../results/circles{i}.jpg', dpi=1000)
    c2 = show_correspondence_lines(image1, image2,
                                   x1[matches[:num_pts_to_visualize, 0]],
                                   y1[matches[:num_pts_to_visualize, 0]],
                                   x2[matches[:num_pts_to_visualize, 1]],
                                   y2[matches[:num_pts_to_visualize, 1]])
    plt.figure()
    plt.imshow(c2)
    plt.savefig(f'../results/lines{i}.jpg', dpi=1000)
    print("result saved")
