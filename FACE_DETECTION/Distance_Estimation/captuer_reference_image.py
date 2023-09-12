import numpy as np
import mediapipe as mp
import cv2 as cv
import utils
from utils import FPS


# face detector function
def detect_face(frame):
    # convert the frame from BGR to RGB
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # process the rgb image to get face points
    results = face_detector.process(rgb_frame)

    # get the width and height of image
    img_height, img_width = frame.shape[:2]

    # create the empty list to hold faces data.
    faces = []

    # check if face found or not
    if results.detections:
        # loop through detections
        for detection in results.detections:
            # get the score/confidence of face
            score = detection.score

            # get face rect and convert into pixel coordinates
            face_rect = np.multiply(
                [
                    detection.location_data.relative_bounding_box.xmin,
                    detection.location_data.relative_bounding_box.ymin,
                    detection.location_data.relative_bounding_box.width,
                    detection.location_data.relative_bounding_box.height,
                ],
                [img_width, img_height, img_width, img_height],
            ).astype(int)

            # create the dict of data
            face_dict = {
                "box": face_rect,
                "score": score[0] * 100,
            }

            faces.append(face_dict)
    return faces


# create the object for face detection
map_face_detection = mp.solutions.face_detection

# camera object
cap = cv.VideoCapture(1)
calc_fps = FPS()

# define the fonts
fonts = cv.FONT_HERSHEY_PLAIN
image_counter = 0
# configure the Face detection model parameters
with map_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6,
) as face_detector:
    while cap.isOpened():
        ret, frame = cap.read()
        image = frame.copy()
        if not ret:
            break
        calc_fps.get_frame_rate(frame)

        # detecting face in frame
        faces = detect_face(frame)

        # check if face found or not
        if faces is not None:
            for face in faces:
                score = face["score"]
                box = face["box"]
                x, y, w, h = box
                utils.text_with_background(
                    frame,
                    f"Score: {score:.2f}",
                    (x, y - 10),
                    fonts,
                    color=(0, 255, 255),
                )
                utils.rect_corners(frame, box, (0, 255, 255), th=3)
                utils.text_with_background(
                    frame,
                    f"press c to capture the image image {image_counter}",
                    (40, 70),
                    fonts,
                    color=(0, 255, 255),
                )
                utils.text_with_background(
                    frame,
                    f"images captured: {image_counter}",
                    (40, 90),
                    fonts,
                    color=(0, 255, 255),
                )
        cv.imshow("frame", frame)
        key = cv.waitKey(1)
        if key == ord("c"):
            image_counter += 1
            # if we press c on keyboard, the image will be save.
            cv.imwrite(f"reference_image{image_counter}.png", image)
        if key == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
