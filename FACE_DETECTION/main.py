import time
import cv2 as cv
import utils
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection

cap = cv.VideoCapture(1)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detector:
    frame_counter = 0
    fonts = cv.FONT_HERSHEY_PLAIN
    start_time = time.time()
    while True:
        frame_counter += 1
        ret, frame = cap.read()
        if ret is False:
            break
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = face_detector.process(rgb_frame)
        frame_height, frame_width, c = frame.shape
        if results.detections:
            for face in results.detections:
                face_react = np.multiply(
                    [
                        face.location_data.relative_bounding_box.xmin,
                        face.location_data.relative_bounding_box.ymin,
                        face.location_data.relative_bounding_box.width,
                        face.location_data.relative_bounding_box.height,
                    ],
                    [frame_width, frame_height, frame_width, frame_height],
                ).astype(int)
                # print(face_react)

                utils.rect_corners(frame, face_react, utils.MAGENTA, th=3)
                utils.text_with_background(
                    frame,
                    f"score: {(face.score[0]*100):.2f}",
                    face_react[:2],
                    fonts,
                    color=utils.MAGENTA,
                    scaling=0.7,
                )
                # print(face.location_data.relative_bounding_box)
        fps = frame_counter / (time.time() - start_time)
        utils.text_with_background(frame, f"FPS: {fps:.2f}", (30, 30), fonts)
        cv.imshow("frame", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
