import cv2
import numpy as np

img_path = '../Ours/Exp_grasping/Exp_chessBoard/'
img_l = cv2.imread(img_path + 'left_image_01.jpg')
img_r = cv2.imread(img_path + 'right_image_01.jpg')

cameraMatrix1 = [[739.679890563678, 0, 308.8957210082277],
                 [0, 756.2991569563665, 249.9280444902564],
                 [0, 0, 1]]
cameraMatrix1 = np.array(cameraMatrix1)
distCoeffs1 = [-0.2550007112875035, -0.2737866460820568, 0.004003127992458922, -0.003989584107380743, 5.431778424896637]
distCoeffs1 = np.array(distCoeffs1)

cameraMatrix2 = [[744.8072871546212, 0, 390.2587529404848],
                 [0, 762.1561343161522, 251.8411492086125],
                 [0, 0, 1]]
cameraMatrix2 = np.array(cameraMatrix2)
distCoeffs2 = [-0.3932230020162802, 0.5645748843538952, 0.007293980916057105, -0.01044972472768517, -0.1745091070334453]
distCoeffs2 = np.array(distCoeffs2)

R = [[0.9999714555927591, -0.006568058635375512, -0.0037347831879257],
     [0.00657232686916477, 0.9999777616281025, 0.001131710552521682],
     [0.003727266991160962, -0.001156224464411338, 0.9999923852838903]]
T = [-5.554172085942337, -0.1523104334826096, 0.4562417072965379]
R = np.array(R)
T = np.array(T)

imageSize = np.shape(img_l)[0:2]
# print(type(cameraMatrix1))
# print(T)

rotation1, rotation2, proj1, proj2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, alpha = 1)
print(Q)
# distort images
undistort_map1, rectify_map1 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, rotation1, proj1, imageSize, cv2.CV_32FC1)
undistort_map2, rectify_map2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, rotation2, proj2, imageSize, cv2.CV_32FC1)

img_l = cv2.remap(img_l, undistort_map1, rectify_map1, cv2.INTER_LINEAR)
img_r = cv2.remap(img_r, undistort_map2, rectify_map2, cv2.INTER_LINEAR)
cv2.imshow('img', img_r)
cv2.waitKey(3000)


# print(rotation1)
# print(rotation2)
# print(proj1)
# print(proj2)
# print(validPixROI1)
# print(validPixROI2)
# img_l = cv2.undistort(img_l, cameraMatrix1, distCoeffs1)
# img_r = cv2.undistort(img_r, cameraMatrix2, distCoeffs2)

gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

ret_l, corners_l = cv2.findChessboardCorners(gray_l, (8, 11), None)
ret_r, corners_r = cv2.findChessboardCorners(gray_r, (8, 11), None)




criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

while True:
    corners_l_ = cv2.cornerSubPix(gray_l, corners_l, (6, 6), (-1, -1), criteria)
    corners_r_ = cv2.cornerSubPix(gray_r, corners_r, (6, 6), (-1, -1), criteria)

    img_l = cv2.drawChessboardCorners(img_l, (8, 11), corners_l_, ret_l)

    img_l = cv2.circle(img_l, (int(corners_l_[32, 0][0]), int(corners_l_[32, 0][1])), radius=15, color=(255, 255, 255), thickness=2)

    cv2.imshow('img', img_l)
    if cv2.waitKey(7) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# print(corners_l_[0, 0][0])
# print(corners_l_[0, :][0])

points4D = cv2.triangulatePoints(proj1, proj2, corners_l_[0, 0], corners_r_[0, 0])
print(points4D)
print(type(points4D))
points4D_ = points4D/points4D[3]
print(points4D_)

# if cv2.waitKey(7) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()