import time
import cv2 as cv
import utils
import mediapipe as mp
import numpy as np
import collections

track_points = collections.deque(maxlen=14)
counter = 0
MOVE_THRESHOLD = 8

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
                fx, fy, _, _ = face_react

                track_points.appendleft([fx, fy])
                counter += 1
                # looping through all the points in collection
                for i in range(1, len(track_points)):
                    if counter >= 10 and i == 1 and track_points[-10] is not None:
                        dX = track_points[-10][0] - track_points[i][0]
                        dY = track_points[-10][1] - track_points[i][1]
                        direction_x, direction_y = "", ""
                        if np.abs(dX) >= MOVE_THRESHOLD:
                            direction_x = "Left" if np.sign(dX) == 1 else "right"
                            cv.arrowedLine(
                                frame,
                                (fx, fy),
                                (fx - int(dX), fy),
                                (0, 255, 0),
                                2,
                                cv.LINE_AA,
                            )
                        if np.abs(dY) >= MOVE_THRESHOLD:
                            direction_y = "Up" if np.sign(dY) == 1 else "Down"
                            cv.arrowedLine(
                                frame,
                                (fx, fy),
                                (fx, fy - int(dY)),
                                (0, 0, 255),
                                2,
                                cv.LINE_AA,
                            )

                        if direction_x == "" and direction_y == "":
                            direction = "stable"
                        else:
                            direction = f"{direction_x} {direction_y}"
                        utils.text_with_background(
                            frame,
                            f"direction: {direction}",
                            (30, 72),
                            cv.FONT_HERSHEY_PLAIN,
                            color=(0, 255, 0),
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
