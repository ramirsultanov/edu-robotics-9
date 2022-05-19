import cv2
import numpy as np
import cv2 as cv

MIN_MATCH_COUNT = 7
img1 = cv.imread('resources/1.png')
img2 = cv.imread('resources/2.png')
imgA = cv.imread('resources/3.jpg')
height2, width2 = img2.shape[:2]
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    imgA = cv.resize(imgA, (img1.shape[0], img1.shape[1]))
    heightA, widthA = imgA.shape[:2]
    img3 = cv.warpPerspective(imgA, M, (width2, height2))
    cv.imshow('img3', img3)
    # mask2 = np.zeros(img3.shape, dtype=np.uint8)
    # channel_count2 = img2.shape[2]
    # ignore_mask_color2 = (255,)*channel_count2
    # p0 = (int(dst[0][0][0]), int(dst[0][0][1]))
    # p1 = (int(dst[1][0][0]), int(dst[1][0][1]))
    # p2 = (int(dst[2][0][0]), int(dst[2][0][1]))
    # p3 = (int(dst[3][0][0]), int(dst[3][0][1]))
    # s = np.array([[int(dst[0][0][0]), int(dst[0][0][1])], [int(dst[1][0][0]), int(dst[1][0][1])], [int(dst[2][0][0]), int(dst[2][0][1])], [int(dst[3][0][0]), int(dst[3][0][1])]])
    # s = np.int32(dst)
    # print(s)
    # cv.fillConvexPoly(mask2, s, ignore_mask_color2)
    # cv.imshow('mask2', mask2)
    # mask2 = cv.bitwise_not(mask2)
    # masked_image2 = cv.bitwise_and(img2, mask2)
    # final = cv.bitwise_or(img3, masked_image2)
    # cv.imshow('final.png', final)
    # print(np.int32(dst))
    empty = np.zeros(img2.shape, dtype=np.uint8)
    # mask2 = cv2.polylines(empty, [np.int32(dst)], True, (255,255,255))
    cv.fillConvexPoly(empty, np.int32(dst), (255, 255, 255))
    empty = cv.bitwise_not(empty)
    cut = cv.bitwise_and(img2, empty)
    final = cv.bitwise_or(img3, cut)
    # cv.imshow('mask2', mask2)
    cv.imshow('empty', empty)
    cv.imshow('cut', cut)
    cv.imshow('final', final)
    cv.waitKey(10000)
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)
