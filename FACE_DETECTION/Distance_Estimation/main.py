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


# focal length finder function
def focal_length_finder(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value


# distance estimation function
def distance_finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance


# create the object for face detection
map_face_detection = mp.solutions.face_detection

# camera object
cap = cv.VideoCapture(1)
calc_fps = FPS()
# variables
# distance from camera to object(face) measured
KNOWN_DISTANCE = 76.2  # centimeter
# width of face in the real world or Object Plane
KNOWN_FACE_WIDTH = 14.3  # centimeter

# define the fonts
fonts = cv.FONT_HERSHEY_PLAIN

ref_image = cv.imread("./Ref_image.png")

# configure the Face detection model parameters
with map_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6,
) as face_detector:
    face_data = detect_face(ref_image)

    # get the width of face in reference image
    rx, ry, rw, rh = face_data[0]["box"]
    # print(rw)
    # calculate the focal length from reference image
    focal_point = focal_length_finder(KNOWN_DISTANCE, KNOWN_FACE_WIDTH, rw)

    while cap.isOpened():
        ret, frame = cap.read()
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
                distance = distance_finder(focal_point, KNOWN_FACE_WIDTH, w)
                utils.text_with_background(
                    frame,
                    f"Distance: {distance:.1f} cm",
                    (x, y - 10),
                    fonts,
                    color=(0, 255, 255),
                )
                utils.rect_corners(frame, box, (0, 255, 255), th=3)
        cv.imshow("frame", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
