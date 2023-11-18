import cv2
import argparse
import numpy as np

def load_model_weights(model_version):
    if model_version == "yolov3":
        weights_file = "weights/yolov3.weights"
        cfg_file = "cfg/yolov3.cfg"
        output_layers = ["yolo_82", "yolo_94", "yolo_106"]
    elif model_version == "yolov4":
        weights_file = "weights/yolov4.weights"
        cfg_file = "cfg/yolov4.cfg"
        output_layers = ["yolo_139", "yolo_150", "yolo_161"]
    elif model_version == "yolov3-tiny":
        weights_file = "weights/yolov3-tiny.weights"
        cfg_file = "cfg/yolov3-tiny.cfg"
        output_layers = ["yolo_16", "yolo_23"]
    elif model_version == "yolov4-tiny":
        weights_file = "weights/yolov4-tiny.weights"
        cfg_file = "cfg/yolov4-tiny.cfg"
        output_layers = ["yolo_30", "yolo_37"]
    else:
        raise ValueError("Invalid YOLO model version specified.")

    net = cv2.dnn.readNet(weights_file, cfg_file)
    layer_names = net.getLayerNames()
    return net, output_layers

# Argument Parser
parser = argparse.ArgumentParser(description='YOLO Object Detection')
parser.add_argument('--model', choices=['yolov3', 'yolov4', 'yolov3-tiny', 'yolov4-tiny'],
                    default='yolov4-tiny', help='Select YOLO model version')
args = parser.parse_args()

# Set confidence threshold
conf_threshold = 0.5

# Load class names
object_names = "object-names/coco.names"
classes = []
with open(object_names, "r") as f:
    classes = f.read().strip().split("\n")

# Load YOLO model based on selected version
net, output_layers = load_model_weights(args.model)

# Open a video stream or capture from a webcam
cap = cv2.VideoCapture(0)  # Change this to the video file path if you want to process a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (width, height), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Post-processing to get bounding boxes
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)

    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # Display the frame
    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
