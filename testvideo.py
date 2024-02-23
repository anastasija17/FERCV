import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DNN = "TF"
if DNN == "CAFFE":
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# Učitavanje modela za prepoznavanje emocija
model = load_model('model_file_30epochs.h5')

video = cv2.VideoCapture('video.mp4')

labels_dictionary = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Neutral', 6: 'Surprise'}
emotion_count = {label: 0 for label in labels_dictionary.values()}

total_frames = 0
frame_skip = 50
frame_rate = 30
seconds_per_interval = 30
confidence_threshold = 0.3

emotion_intervals = []
emotions = list(labels_dictionary.values())

# Povećanje faktora skaliranja i dimenzija slike
scale_factor = 1.5
new_width, new_height = 1000, 800

def detect_faces(frame):
    h, w = frame.shape[:2]

    resized_frame = cv2.resize(frame, (new_width, new_height))
    blob = cv2.dnn.blobFromImage(resized_frame, scale_factor, (new_width, new_height), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x, y, x_end, y_end = box.astype(int)

            if x >= 0 and y >= 0 and x_end >= 0 and y_end >= 0:
                faces.append((x, y, x_end, y_end))

    return faces

while True:
    ret, frame = video.read()
    if not ret:
        break

    total_frames += 1
    if total_frames % (frame_rate * seconds_per_interval) == 0:
        total_count = sum(emotion_count.values())
        emotion_percentages = {label: (count / total_count) * 100 for label, count in emotion_count.items()}
        emotion_intervals.append([emotion_percentages[label] for label in emotions])
        emotion_count = {label: 0 for label in labels_dictionary.values()}

    if total_frames % frame_skip != 0:
        continue

    faces = detect_faces(frame)

    for (x, y, x_end, y_end) in faces:
        face = frame[y:y_end, x:x_end]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(face_gray, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        emotion = labels_dictionary[label]
        emotion_count[emotion] += 1

        cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

x = np.arange(len(emotions))

def update(frame):
    interval_start = frame * seconds_per_interval
    interval_end = (frame + 1) * seconds_per_interval
    interval_label = f'Interval {interval_start}-{interval_end}s'
    
    plt.clf()
    plt.bar(x, emotion_intervals[frame], color=['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan'])
    plt.xlabel('Emotions')
    plt.ylabel('Percentage')
    plt.title(f'Emotion Percentages - {interval_label}')
    plt.xticks(x, emotions)
    plt.ylim(0, 100)

ani = FuncAnimation(plt.gcf(), update, frames=len(emotion_intervals), repeat=False, interval=10000)
plt.show()

video.release()
cv2.destroyAllWindows()
