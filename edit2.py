import os
import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Set the page configuration
st.set_page_config(layout="wide")

# Set the image path
image_path = 'MathGestures.png'

# Check if the image exists
if os.path.exists(image_path):
    st.image(image_path)
else:
    st.error(f"Image file not found: {image_path}")

# Create columns for layout
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

# Configure the Generative AI model
genai.configure(api_key="AIzaSyAu7w2tMO4kIAiB-RDMh8vywmF8OqBjpQk")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)  # Change to 0 or another index if necessary
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    # Find hands in the current frame
    hands, img = detector.findHands(img, draw=False, flipType=True)

    # Check if any hands are detected
    if hands:
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        fingers = detector.fingersUp(hand)  # Count the number of fingers up for the first hand
        print(fingers)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None

    # Draw if the index finger is up
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]  # Index finger tip position
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    # Reset the canvas if only the thumb is up
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)

    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  # Check if all fingers except the thumb are up
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve the math problem and give explaination too", pil_image])
        return response.text

prev_pos = None
canvas = None
image_combined = None
output_text = ""

# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    success, img = cap.read()
    
    # Check if the frame was successfully captured
    if not success:
        st.error("Failed to capture image from webcam.")
        break  # Exit the loop if the frame capture fails

    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    # Keep the window open and update it for each frame
    cv2.waitKey(1)

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
