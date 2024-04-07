import cv2
import mediapipe as mp

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Error: Could not read from webcam")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Initialize finger states
            thumb_raised = False
            index_raised = False
            middle_raised = False
            ring_raised = False
            little_raised = False

            # Thumb
            if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y:
                thumb_raised = True

            # Index finger
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
                index_raised = True

            # Middle finger
            if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
                middle_raised = True

            # Ring finger
            if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
                ring_raised = True

            # Little finger
            if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
                little_raised = True

            if not index_raised and not middle_raised and not ring_raised and not little_raised:
                print("Fist clenched")
            elif index_raised and middle_raised and ring_raised and little_raised:
                print("All fingers")
            else:
                raised_fingers = sum([index_raised, middle_raised, ring_raised, little_raised])
                print(f"{raised_fingers} finger(s) up")

            # Display hand landmarks
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", img)
    cv2.waitKey(500)
    
    # Check for window close event or 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
