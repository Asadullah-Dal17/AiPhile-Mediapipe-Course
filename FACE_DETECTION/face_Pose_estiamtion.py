import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    # it will be faster but the image will be read only because of this
    # What it does? It will not create a copy of the image
    image.flags.writeable = False

    # get the results
    results = face_mesh.process(image)

    # improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            for idx, lm in enumerate(face_landmarks.landmark):
                if (
                    idx == 33
                    or idx == 263
                    or idx == 1
                    or idx == 61
                    or idx == 291
                    or idx == 199
                ):
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # convert to np
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # the cara matrix
            focal_length = 1 * img_w

            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )

            # the distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix
            )

            # get the rotation matrix
            rot_mat, jac = cv2.Rodrigues(rot_vec)

            # get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rot_mat)

            # get the rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(
                nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix
            )

            p1 = (int(nose_2d[0]), int(nose_2d[1]))  # ponto inicial da linha
            # ponto final da linha, que é o ponto inicial projetado de acordo com os angulos
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (0, 255, 0), 2)
            cv2.putText(
                image,
                f"X: {int(x)}",
                (20, 110),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (0, 255, 0),
                3,
            )
            cv2.putText(
                image,
                f"Y: {int(y)}",
                (20, 140),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (0, 255, 0),
                3,
            )
            cv2.putText(
                image,
                f"Z: {int(z)}",
                (20, 170),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (0, 255, 0),
                3,
            )

    end = time.time()
    total_time = end - start
    fps = 1 / total_time
    cv2.putText(
        image, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3
    )
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
