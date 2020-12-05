# A simple demo for AR 

AR Demo Based on the **ORB** feature extraction. in this demo, we show how to obtain the plane information of a specified object and project a 3D object on it.




## Preparation

To detect the plane, we must first prepare a plane (a picture), and then find the **feature points** of the picture in the video stream to achieve tracking.

To achieve the detection of the **predetermined plane**, the most direct method is the feature points, such as ORB features, corner features, etc. The extraction of these features is nothing more than the shape and optical flow of each pixel to determine whether they are similar. The matching points, and these algorithms are a bit complicated to implement. However, we can easily implement it with **opencv**. It's **opencv-python** in python language.

First prepare a template picture (placed in the reference folder). This picture is the plane to be detected. For a more stable detection, more features can be added to the photo.

The next step is to call the camera, and then detect the template picture just now in the video stream, and perform feature point detection and correspondence.



```python
import cv2
import numpy as np
# 调用摄像头设备
device = 0
cap =cv2.VideoCapture(device)

min_matches = 15
model = cv2.imread('model.jpg', 0)
print(model.shape)
while True:
    res, frame = cap.read(0)
    if res:
        cv2.imshow('cap', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
```



## Feature Matching

Next, let's finish a very simple task to find out the pictures from the video stream. Simply put, it is to do **feature matching**. In opencv, there is a class of `cv2.ORB_create()`, which can create an ORB feature extractor. Through feature extraction, the corresponding feature points can be obtained, and the matching can be performed by the matcher.

The core steps are as follows:

```python
model = cv2.imread('reference/model.jpg', 0)
device = 0
cap = cv2.VideoCapture(device)

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cv2.resizeWindow("camera", size[0], size[1])
while True:
    # read the current frame
    ret, frame = cap.read(0)
    if ret:
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        kp_model, des_model = orb.detectAndCompute(model, None)
        kp_frame, des_frame = orb.detectAndCompute(frame, None)

        matches = bf.match(des_model, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        res = cv2.drawMatches(model, kp_model, frame, kp_frame,
                              matches[: MIN_MATCHES], 0, flags=2)
        cv2.imshow('res', res)
        cv2.waitKey(0)
```
After matching by `BFMatcher`, our matching here is done using Hamming distance. If you want to study the principles used here, you can also look at some introductions about Hamming distance.

If you run it, you should be able to see the feature matching image (res.jpg).
![res](res.jpg)



## Object Recognition

This is not enough. We'd better mark the target with a box to facilitate our subsequent planar mapping. Simply put, this step is the most difficult part. We need to find the mapping relationship with the target image based on the template image, and then determine a conversion matrix. This operation is called **Homography**.

```python
src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
# compute Homography
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
h, w = model.shape[0], model.shape[1]
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
# project corners into frame
dst = cv2.perspectiveTransform(pts, M)  
# connect them with lines
img2 = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
```

The above steps are done on the basis of detecting the key points of the template picture and the key points of each frame. The next thing to realize is how to put our 3D model on the plane.

## Last Step

Let's finish the task.

here are two problems to be solved in this step:

1. According to the homography matrix and external camera parameters, obtain the corrected coordinate system, and place the OBJ stereo model on it;
2. How to resolve OBJ?

![](https://bitesofcode.files.wordpress.com/2017/09/selection_003.png)

As shown in the figure above, under normal circumstances, the external parameters of the camera will have four parameters. Three coordinates of the three-dimensional coordinate system, and a rotation angle. We can remove z, that is, remove R3, because we are going to project the OBJ into a plane, so z will naturally be 0.

![](https://bitesofcode.files.wordpress.com/2018/07/selection_017.png)

Then came a very complicated operation. In short, we directly use the code to show the final effect:

```python
def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography *= -1
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)
```

Finally, you can place a 3D object on your target object from the video stream.


## Code for Use

Replace your template picture with the picture in the reference folder. In addition, please make sure that there is a camera device available on your computer.

run:

```python
python main.py
```

