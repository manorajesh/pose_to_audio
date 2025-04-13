from pythonosc import udp_client
from tracker import HandTracker
from sys import argv
import cv2

# Configure the OSC client
ip = "127.0.0.1"
port = 7001
client = udp_client.SimpleUDPClient(ip, port)

# Function to send test OSC messages


def send_test_messages(tracker):
    print(f"Sending OSC messages to {ip}:{port}")
    prevAvgCurl = 0
    prevHandsDistance = 0
    prevDeltaV = 0
    prevPosition = 0
    smoothing = 0.3

    while True:
        result = tracker.getHands()
        if result is None:
            break

        hands, frame = result
        if hands is not None:
            avgCurl = 0
            handsDistance = 0
            deltaV = 0
            yPos = []
            for hand in hands:
                if hand.detected:
                    for idx, (_, curl) in enumerate(hand.finger_curls.items()):
                        avgCurl += curl

                    avgCurl /= 5

                    value = hand.deltaV
                    deltaV += value

                    value = tracker.handsDistance if tracker.handsDistance is not None else 0
                    handsDistance += value

                    yPos.append(hand.smoothedY)

            avgCurl /= 2
            deltaV /= 2
            handsDistance /= 2
            yPos = min(yPos or [0])

            # Apply smoothing
            avgCurl = prevAvgCurl + (avgCurl - prevAvgCurl) * smoothing
            deltaV = prevDeltaV + (deltaV - prevDeltaV) * smoothing
            handsDistance = prevHandsDistance + \
                (handsDistance - prevHandsDistance) * smoothing
            yPos = prevPosition + (yPos - prevPosition) * smoothing

            # Update previous values
            prevAvgCurl = avgCurl
            prevDeltaV = deltaV
            prevHandsDistance = handsDistance
            prevPosition = yPos

            message = "/ch/1"
            value = avgCurl / 3
            # print(f"curl: \t\t\t\t{'*' * int(value)}")
            client.send_message(message, value)

            message = "/ch/2"
            value = deltaV / 10
            # print(f"deltaV: \t\t\t{'*' * int(value)}")
            client.send_message(message, value)

            message = "/ch/3"
            value = handsDistance / 100
            # print(f"hands distance: \t\t{'*' * int(value)}")
            client.send_message(message, value)

            message = "/ch/4"
            value = 10 - yPos / 100
            print(f"yPos: \t\t\t{'*' * int(value)}")
            client.send_message(message, value)

        # Display the frame
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.close()
    cv2.destroyAllWindows()


# Run the test
if __name__ == "__main__":
    # Get video file path from command-line arguments
    video_source = argv[1] if len(argv) > 1 else 0
    tracker = HandTracker(video_source=video_source)
    send_test_messages(tracker)
