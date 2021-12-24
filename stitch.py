"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""

import cv2
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random
# from tqdm import tqdm
from math import floor, ceil


def solution(left_img, right_img, feature_method = 'sift', sift_features = 1000,  ratio = 0.75):
    """
    :param left_img:
    :param right_img:
    :param feature_method: descriptor method to be used
    :param sift_features: if 'sift' max features to be used
    :param ratio: \eta for ratio testing
    :return: final_img
    """

    print('Finding keypoints using '+feature_method+':')
    if(feature_method=='orb'):
        kpdes = cv2.ORB_create()
    if(feature_method=='sift'):
        kpdes = cv2.SIFT_create(sift_features)

    lkp, ldes = kpdes.detectAndCompute(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), None)
    rkp, rdes = kpdes.detectAndCompute(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY), None)

    print('-- Keypoints and descriptors found.')

    print('Finding matching keypoint pairs using descriptors: ')

    print('-- KNN and Ratio testing')

    pairs = []

    for i in range(len(lkp)):
        dist = np.sqrt(np.sum(np.square(ldes[i]-rdes),axis=1))
        rank = np.argsort(dist)
        index1 = rank[0]
        index2 = rank[1]

        if(dist[index1]<ratio*dist[index2]):
            pairs.append([i, index1])

    print('-- Cross checking')
    new_pairs = []

    for pair in pairs:
        r_ind = pair[1]
        dist = np.sqrt(np.sum(np.square(rdes[r_ind]-ldes),axis=1))
        rank = np.argsort(dist)
        index1 = rank[0]

        if(index1==pair[0]):
            new_pairs.append(pair)

    pt_pairs = [[lkp[x[0]].pt, rkp[x[1]].pt] for x in new_pairs]
    pt_pairs = np.array(pt_pairs)

    def make_matrix(pairs, n):

        A = np.zeros((2*n, 9))

        for i in range(n):
            x_, y_ = pairs[i][0]
            x, y = pairs[i][1]
            row1 = np.array([x, y,  1, 0, 0, 0, -1*x_*x, -1*x_*y, -1*x_])
            row2 = np.array([0, 0,  0, x, y, 1, -1*y_*x, -1*y_*y, -1*y_])
            A[2*i] = row1
            A[2*i+1] = row2
        return A

    print('-- Keypoint Matching Completed.')

    print('RANSAC in progress to find homography matrix: ')

    np.random.seed(50)

    t = 5

    k = 5000

    inliers = []
    best_selection = None
    best_H = None

    for i in range(k):
        idx = np.random.choice(len(pt_pairs), 4, replace=False)
        p = pt_pairs[idx]
        A = make_matrix(p, 4)
        # A
        U, S, V_T = np.linalg.svd(A)
        H = V_T[-1].reshape((3,3))
        H = H/H[-1,-1]
        # H
        tmp = []
        for i in range(len(pt_pairs)):
            if(i not in idx):
                p_i_ = np.array([pt_pairs[i][0][0], pt_pairs[i][0][1], 1])
                p_i = np.array([pt_pairs[i][1][0], pt_pairs[i][1][1], 1])
                m = np.matmul(H, p_i)
                m = m/m[2]
                # print(m, p_i_)
                d = np.sqrt(np.sum(np.square(m-p_i_)))
                if(d<t):
                    d = np.sqrt(np.sum(np.square(m-p_i_)))
                    tmp.append([p_i_[:2], p_i[:2]])
        if(len(tmp)>len(inliers)):
            inliers = np.array(tmp)
            best_selection = p
            best_H = H

    print('-- Final Homography Matrix Calculation:')
    S = np.concatenate((inliers, best_selection))
    A = make_matrix(S, S.shape[0])
    # A
    U, S, V_T = np.linalg.svd(A)
    H = V_T[-1].reshape((3,3))
    final_H = H/H[-1,-1]

    print('-- Final Homography Matrix Calculated')

    print('Stitching in progress:')

    print('-- Calculating x and y bounds:')
    xs = []
    ys = []
    for y in range(right_img.shape[0]):
        for x in range(right_img.shape[1]):
            pt = np.array([x,y,1])
            m = np.matmul(final_H, pt)
            m = m/m[2]
            xs.append(m[0])
            ys.append(m[1])
    
    max_x, max_y = ceil(max(xs)), ceil(max(ys))

    if(min(xs)<0):
        min_x = floor(min(xs))
    else:
        min_x = ceil(min(xs))
    if(min(ys)<0):
        min_y = floor(min(ys))
    else:
        min_y = ceil(min(ys))
    
    h, w = max([max_x, left_img.shape[0], right_img.shape[0]])-min([0, min_x]), \
            max([max_y, left_img.shape[1], right_img.shape[1]])-min([0, min_y])

    print('-- Generating translation matrix for complete visuals: ')
    
    x_offset = -1*min([0, min_x])
    y_offset = -1*min([0, min_y])

    offset = np.array([[ 1 , 0 , x_offset],
    [ 0 , 1 , y_offset+1],
    [ 0 , 0 ,    1    ]])

    # np.matmul(offset, final_H)
    print('-- Warping and Joining the images')
    final_image = cv2.warpPerspective(right_img, np.matmul(offset, final_H),(h, w))
    final_image[y_offset:y_offset+left_img.shape[0], x_offset:x_offset+left_img.shape[1]] = left_img

    print('Stitching complete!')

    return final_image
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('result.jpg', result_img)