import cv2
import tensorflow as tf
import numpy as np
import time

# Load model
model = tf.keras.models.load_model("facialemotionmodel.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

total_frames = 0
attention_frames = 0
last_expression = None
start_time = time.time()
expression_times = {label: 0 for label in labels.values()}

while True:
    ret, im = webcam.read()
    if not ret:
        break
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    if len(faces) > 0:
        attention_frames += 1

    total_frames += 1

    try:
        for (p, q, r, s) in faces:
            image = gray[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, prediction_label, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            if last_expression != prediction_label:
                if last_expression is not None:
                    expression_times[last_expression] += time.time() - start_time
                start_time = time.time()
                last_expression = prediction_label
        cv2.imshow("Output", im)

        # Check for specific key press (e.g., 'q') to stop the camera feed
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    except cv2.error:
        pass

# Calculate time spent on each expression
total_time = time.time() - start_time
expression_percentages = {label: (time / total_time) * 100 for label, time in expression_times.items()}

# Calculate attention percentage based on time spent on happy or neutral expressions
attention_percentage = (expression_percentages['happy'] + expression_percentages['neutral'])
inattention_percentage = 100 - attention_percentage

print("Attention Percentage:", attention_percentage)
print("Inattention Percentage:", inattention_percentage)

webcam.release()
cv2.destroyAllWindows()
