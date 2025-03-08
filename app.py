import os
import cv2
from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load trained models
person_model = YOLO("weights/person_detection.pt")
ppe_model = YOLO("weights/ppe_detection.pt")

# Define folders
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
CROPPED_FOLDER = "static/cropped"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

# Define PPE class names
PPE_CLASSES = ["hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Perform inference
    result_image = detect_ppe(file_path)
    
    return render_template("result.html", image=result_image)

def detect_ppe(image_path):
    image = cv2.imread(image_path)
    original_image = image.copy()

    # Step 1: Detect Persons
    person_results = person_model(image)
    cropped_persons = []

    for result in person_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            # Draw bounding box around person
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(original_image, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Crop person
            cropped_image = image[y1:y2, x1:x2].copy()
            cropped_path = os.path.join(CROPPED_FOLDER, f"person_{idx}.jpg")
            cv2.imwrite(cropped_path, cropped_image)
            cropped_persons.append((cropped_image, (x1, y1)))

    # Step 2: Detect PPE on Cropped Persons
    for cropped_image, (x1, y1) in cropped_persons:
        ppe_results = ppe_model(cropped_image)

        for ppe_result in ppe_results:
            ppe_boxes = ppe_result.boxes.xyxy.cpu().numpy()
            ppe_classes = ppe_result.boxes.cls.cpu().numpy()

            for i, ppe_box in enumerate(ppe_boxes):
                px1, py1, px2, py2 = map(int, ppe_box)

                # Adjust PPE coordinates to full image
                px1 += x1
                py1 += y1
                px2 += x1
                py2 += y1

                class_id = int(ppe_classes[i])
                label = PPE_CLASSES[class_id] if class_id < len(PPE_CLASSES) else "PPE"

                # Draw PPE bounding box
                cv2.rectangle(original_image, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(original_image, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save the output image
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, original_image)

    return os.path.basename(result_path)

@app.route("/results/<filename>")
def get_result_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
