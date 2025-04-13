import cv2
import mediapipe as mp
import numpy as np


class Hand:
    def __init__(self, label):
        self.label = label
        self.smoothedX = None
        self.smoothedY = None
        self.prevSmoothedX = None
        self.prevSmoothedY = None
        self.deltaX = 0.0
        self.deltaY = 0.0
        self.deltaV = 0.0
        self.detected = False
        self.finger_curls = {
            "thumb": 0.0,
            "index": 0.0,
            "middle": 0.0,
            "ring": 0.0,
            "pinky": 0.0
        }
        self.prevFingerCurls = {
            "thumb": 0.0,
            "index": 0.0,
            "middle": 0.0,
            "ring": 0.0,
            "pinky": 0.0
        }
        self.distance = None
        self.avgCurl = 0.0


class HandTracker:
    def __init__(self, video_source=0, smoothing=0.5, frame_width=640, frame_height=480):
        self.videoCap = cv2.VideoCapture(video_source)

        handSolution = mp.solutions.hands
        self.hands = handSolution.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
        )

        self.leftHand = Hand('Left')
        self.rightHand = Hand('Right')
        self.smoothing = smoothing
        self.handsDistance = None
        self.prevHandsDistance = None
        self.prevDeltaV = None

    def getHands(self):
        success, img = self.videoCap.read()
        if not success:
            # Reset to the beginning of the video
            self.videoCap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, img = self.videoCap.read()
            if not success:
                return None, None  # If still unsuccessful, return None

        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        recHands = self.hands.process(imgRgb)

        h, w, _ = img.shape

        # Reset detection status
        self.leftHand.detected = False
        self.rightHand.detected = False

        if recHands.multi_hand_landmarks and recHands.multi_handedness:
            for hand_landmarks, handedness in zip(recHands.multi_hand_landmarks, recHands.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'

                xs = np.array([lm.x for lm in hand_landmarks.landmark])
                ys = np.array([lm.y for lm in hand_landmarks.landmark])

                avgX_norm = np.mean(xs)
                avgY_norm = np.mean(ys)

                # Convert normalized to pixel coordinates
                currentX = int(w - (avgX_norm * w))
                currentY = int(avgY_norm * h)

                # Get the Hand instance corresponding to the label
                if label == 'Left':
                    hand = self.leftHand
                else:
                    hand = self.rightHand

                # Initialize smoothed positions if None
                hand.smoothedX = currentX
                hand.smoothedY = currentY

                # Compute delta velocities
                if hand.prevSmoothedX is not None and hand.prevSmoothedY is not None:
                    hand.deltaX = hand.smoothedX - hand.prevSmoothedX
                    hand.deltaY = hand.smoothedY - hand.prevSmoothedY
                    currentDeltaV = np.sqrt(hand.deltaX**2 + hand.deltaY**2)
                    if self.prevDeltaV is None:
                        hand.deltaV = currentDeltaV
                    else:
                        hand.deltaV = self.prevDeltaV + \
                            (currentDeltaV - self.prevDeltaV) * self.smoothing
                    self.prevDeltaV = hand.deltaV
                else:
                    hand.deltaX = 0.0
                    hand.deltaY = 0.0
                    hand.deltaV = 0.0

                # Update previous smoothed positions
                hand.prevSmoothedX = hand.smoothedX
                hand.prevSmoothedY = hand.smoothedY

                hand.detected = True

                # Calculate hand distance from camera
                zs = np.array([lm.z for lm in hand_landmarks.landmark])
                hand.distance = np.mean(zs)

                # Calculate finger curls
                finger_curls = self.calculateFingerCurls(hand_landmarks)
                for finger in finger_curls:
                    if hand.prevFingerCurls[finger] is None:
                        hand.finger_curls[finger] = finger_curls[finger]
                    else:
                        hand.finger_curls[finger] = hand.prevFingerCurls[finger] + \
                            (finger_curls[finger] -
                             hand.prevFingerCurls[finger]) * self.smoothing
                    hand.prevFingerCurls[finger] = hand.finger_curls[finger]
                hand.avgCurl = np.mean(list(hand.finger_curls.values()))

                # Draw hand landmarks on the image
                mp.solutions.drawing_utils.draw_landmarks(
                    img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        if self.leftHand.detected and self.rightHand.detected:
            dx = self.leftHand.smoothedX - self.rightHand.smoothedX
            dy = self.leftHand.smoothedY - self.rightHand.smoothedY
            currentHandsDistance = np.sqrt(dx**2 + dy**2)
            if self.prevHandsDistance is None:
                self.handsDistance = currentHandsDistance
            else:
                self.handsDistance = self.prevHandsDistance + \
                    (currentHandsDistance - self.prevHandsDistance) * self.smoothing
            self.prevHandsDistance = self.handsDistance

        return (self.leftHand, self.rightHand), img

    def calculateFingerCurls(self, hand_landmarks):
        # Calculate the curl for each finger
        finger_curls = {}
        landmarks = hand_landmarks.landmark

        fingers = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20]
        }

        for finger, idx in fingers.items():
            mcp = landmarks[idx[0]]  # Base joint
            pip = landmarks[idx[1]]  # Proximal interphalangeal joint
            dip = landmarks[idx[2]]  # Distal interphalangeal joint
            tip = landmarks[idx[3]]  # Fingertip

            # Calculate vectors
            vec1 = np.array([pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z])
            vec2 = np.array([dip.x - pip.x, dip.y - pip.y, dip.z - pip.z])
            vec3 = np.array([tip.x - dip.x, tip.y - dip.y, tip.z - dip.z])

            # Calculate angles between vectors
            angle1 = self.angleBetweenVectors(vec1, vec2)
            angle2 = self.angleBetweenVectors(vec2, vec3)

            # Average the angles to get the curl value
            curl = (angle1 + angle2) / 2
            finger_curls[finger] = curl

        return finger_curls

    def angleBetweenVectors(self, v1, v2):
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_v1, unit_v2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return np.degrees(angle)

    def close(self):
        self.videoCap.release()

    def __del__(self):
        self.close()
