import onnxruntime
import numpy as np
import cv2
import time
import mediapipe as mp

idx_to_class = {0: 'Neutral',
                1: 'Happy',
                2: 'Sad',
                3: 'Surprise',
                4: 'Fear',
                5: 'Disgust',
                6: 'Anger',
                7: 'Contempt'}

# Draw Bounding box, headpose
def draw_bbox_axis(frame, face_pos):
    (x, y, x2, y2) = face_pos

    # BBox draw
    cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)),
                  color=(255, 255, 255), thickness=2)

# Draw Russell's Circumplex Model
def draw_russell(frame, valence, arousal, emotion):
    # TODO : x,y 명칭을 반대로했는데 언젠가 수정하자
    x_shape, y_shape, _ = frame.shape
    base_xy = 150
    len_xy = 120

    # Box 1
    add_image = np.zeros((299, y_shape, 3), np.uint8)
    add_image = cv2.rectangle(add_image, (base_xy - len_xy, base_xy - len_xy), (base_xy + len_xy, base_xy + len_xy),
                              color=(255, 255, 255), thickness=2)
    add_image = cv2.line(add_image, (base_xy - len_xy, base_xy), (base_xy + len_xy, base_xy), color=(255, 255, 255),
                         thickness=1)
    add_image = cv2.line(add_image, (base_xy, base_xy - len_xy), (base_xy, base_xy + len_xy), color=(255, 255, 255),
                         thickness=1)
    add_image = cv2.putText(add_image, 'Valence', (base_xy - 42, base_xy - len_xy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (255, 255, 255), 1)
    add_image = cv2.putText(add_image, 'Arousal', (base_xy + len_xy + 5, base_xy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (255, 255, 255), 1)

    valence_xy = int(base_xy + len_xy * valence)
    arousal_xy = int(base_xy - len_xy * arousal)  # Y축이라 마이너스 적용
    add_image = cv2.line(add_image,
                         (valence_xy, arousal_xy),
                         (valence_xy, arousal_xy),
                         color=(0, 0, 255), thickness=5)

    # Box 2
    box2_y_region = 600
    add_image = cv2.putText(add_image, 'Output', (y_shape - box2_y_region, base_xy - len_xy + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (255, 255, 255), 2)
    add_image = cv2.putText(add_image, f'Valence : {str(valence)}', (y_shape - box2_y_region, base_xy - len_xy + 47),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
    add_image = cv2.putText(add_image, f'Arousal : {str(arousal)}', (y_shape - box2_y_region, base_xy - len_xy + 69),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
    add_image = cv2.putText(add_image, f'Emotion : {str(emotion)}', (y_shape - box2_y_region, base_xy - len_xy + 91),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

    # Line
    add_image = cv2.line(add_image, (y_shape//2, 20), (y_shape//2, add_image.shape[1]-20), color=(255, 255, 255), thickness=2)

    frame = cv2.vconcat([frame, add_image])
    frame = cv2.resize(frame, (int(x_shape/frame.shape[0]*y_shape), x_shape))

    return frame

# ONNX 모델 로드
onnx_model_path = "emonet.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)

# 입력 텐서 생성
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.9)
cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 거울 모드
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    loop_start_time = time.time()
    detected = face_detection.process(rgb_img)
    detect_time = time.time() - loop_start_time
    start_time = time.time()

    if detected.detections:
        face_pos = detected.detections[0].location_data.relative_bounding_box
        x = int(rgb_img.shape[1] * max(face_pos.xmin, 0))
        y = int(rgb_img.shape[0] * max(face_pos.ymin, 0))
        w = int(rgb_img.shape[1] * min(face_pos.width, 1))
        h = int(rgb_img.shape[0] * min(face_pos.height, 1))

        # face_pos 확정
        face_plus_scalar = 20
        x2 = min(x + w + face_plus_scalar, rgb_img.shape[1])
        y2 = min(y + h + face_plus_scalar, rgb_img.shape[0])
        x = max(0, x - face_plus_scalar)
        y = max(0, y - face_plus_scalar)


        def get_scale_center(bb):
            center = np.array([bb[2] - (bb[2] - bb[0]) / 2, bb[3] - (bb[3] - bb[1]) / 2])
            scale = (bb[2] - bb[0] + bb[3] - bb[1]) / 220.0

            return scale, center


        def get_transform(center, scale, res, rot=0):
            # Generate transformation matrix

            h = 200 * scale
            t = np.zeros((3, 3))
            t[0, 0] = float(res[1]) / h
            t[1, 1] = float(res[0]) / h
            t[0, 2] = res[1] * (-float(center[0]) / h + .5)
            t[1, 2] = res[0] * (-float(center[1]) / h + .5)
            t[2, 2] = 1

            return t

        bb = [x, y, x2, y2]
        scale, center = get_scale_center(bb)
        aug_rot = 0
        aug_scale = 1
        scale *= aug_scale
        dx, dy = 0, 0
        center[0] += dx * center[0]
        center[1] += dy * center[1]
        mat = get_transform(center, scale, (256, 256), aug_rot)[:2]
        face_img = cv2.warpAffine(rgb_img, mat, (256, 256))  # , borderMode= cv2.BORDER_WRAP)

        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = face_img.astype(np.float32)
        face_img /= 255
        face_img = np.expand_dims(face_img, axis=0)
        outputs = session.run(output_names, {input_name: face_img})

        emotion = idx_to_class[np.argmax(outputs[1])]
        valence = round(outputs[2][0], 2)
        arousal = round(outputs[3][0], 2)

        # Draw Image
        draw_bbox_axis(frame=frame, face_pos=(x, y, x2, y2))
        frame = cv2.putText(frame, f'Emotion : {emotion}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        frame = cv2.putText(frame, f'Valence : {str(valence)}, Arousal : {str(arousal)}',
                            (x, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                            2)
        frame = draw_russell(frame, valence, arousal, emotion)

        # Show Image
        cv2.imshow("", frame)

        print(">>> Use Time : Detect {}, Predict {}".format(round(detect_time,2), round(time.time() - start_time, 2)))

    if cv2.waitKey(1) & 0xff == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
