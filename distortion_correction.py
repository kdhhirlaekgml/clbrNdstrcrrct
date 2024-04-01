import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = r'C:\Users\kdhhi\Desktop\서울과학기술대학교\강의자료\2024 1학기\컴퓨터비전\camera_calibration_sample.mp4'
K = np.array([[654.60440271, 0, 547.56880934],
              [0, 687.54028032, 598.78802921],
              [0, 0, 1]]) # Derived from `calibrate_camera.py`
dist_coeff = np.array([0.03210618, -0.12496054, -0.0005484, 0.00256446, 0.16542982])

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
output_video = cv.VideoWriter('rectified_output.mp4', fourcc, 30.0, (int(video.get(3)), int(video.get(4))))

# Run distortion correction
show_rectify = True
map1, map2 = None, None
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Rectify geometric distortion (Alternative: `cv.undistort()`)
    info = "Original"
    if show_rectify:
        if map1 is None or map2 is None:
            map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, None, (img.shape[1], img.shape[0]), cv.CV_32FC1)
        img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
        info = "Rectified"
    cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Write the frame into the file 'rectified_output.mp4'
    output_video.write(img)

    # Show the image and process the key event
    cv.imshow("Geometric Distortion Correction", img)
    key = cv.waitKey(10)
    if key == ord(' '):     # Space: Pause
        key = cv.waitKey()
    if key == 27:           # ESC: Exit
        break
    elif key == ord('\t'):  # Tab: Toggle the mode
        show_rectify = not show_rectify

# Release everything if job is finished
video.release()
output_video.release()
cv.destroyAllWindows()
