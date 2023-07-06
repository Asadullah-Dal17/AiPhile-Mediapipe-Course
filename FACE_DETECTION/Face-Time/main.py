import time
import cv2 as cv
import utils
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection


class FaceTimeSpend:
    def __init__(self, start_time=time.time()) -> None:
        self.start_time = start_time
        self.session_id = 0
        self.session_time_list = []

    def calculate_session_time(self):
        self.current_session_time = time.time() - self.start_time
        return self.current_session_time

    def update_time(self):
        self.start_time = time.time()
        if self.current_session_time >= 2.5:
            self.session_id += 1
            self.session_time_list.append(self.current_session_time)
            self.current_session_time = 0

    def formate_time(self, seconds):
        return time.strftime("%H:%M:%S", time.gmtime(seconds))

    def get_time(self):
        total_seconds = self.current_session_time + sum(self.session_time_list)
        total_time_formatted = self.formate_time(total_seconds)
        session_time_formatted = self.formate_time(self.current_session_time)
        return total_time_formatted, session_time_formatted, self.session_id


cap = cv.VideoCapture(1)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detector:
    frame_counter = 0
    fonts = cv.FONT_HERSHEY_PLAIN
    start_time = time.time()
    face_timer = FaceTimeSpend()
    while True:
        frame_counter += 1
        ret, frame = cap.read()
        if ret is False:
            break
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = face_detector.process(rgb_frame)
        frame_height, frame_width, c = frame.shape
        if results.detections:
            session_time = face_timer.calculate_session_time()
            print(f"session_time {session_time:.2f}", end="\r")
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
        else:
            face_timer.update_time()
        total_face_time_ft, session_time_ft, session_id = face_timer.get_time()
        # print(face.location_data.relative_bounding_box)
        utils.text_with_background(
            frame,
            f"FT: {total_face_time_ft} ST: {session_time_ft} S-id {session_id}",
            (30, 50),
            fonts,
            color=utils.YELLOW,
        )

        fps = frame_counter / (time.time() - start_time)
        utils.text_with_background(frame, f"FPS: {fps:.2f}", (30, 30), fonts)
        cv.imshow("frame", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
