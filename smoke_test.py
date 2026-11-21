import cv2, mediapipe as mp

hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
    cv2.imshow('smoke test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()