#  AI-Powered Person & PPE Detection System

##  Project Overview
This project is an **AI-powered Person and PPE (Personal Protective Equipment) Detection System** designed to enhance workplace safety. The system can **detect people in an image** and **identify whether they are wearing PPE** (helmets, gloves, masks, safety vests, etc.).  

It uses **YOLOv8** for object detection and **Flask** for deployment, providing a user-friendly web interface.

## Screen Shot

![Screenshot 2025-03-08 122520](https://github.com/user-attachments/assets/118a318d-6e32-4026-bd8f-40f530d1b9a6)

![Screenshot 2025-03-08 123339](https://github.com/user-attachments/assets/497c3297-2637-40bd-ae66-36568bb16477)

##  Features
**Detects persons** in an image  
 **Identifies PPE items** (hard hat, gloves, mask, glasses, boots, vest, PPE suit, ear protector, safety harness)  
 **Processes cropped person images** for accurate PPE detection  
 **User-friendly web interface** with Flask  
 **Real-time image processing** using OpenCV  
 **Fully responsive UI** with animations and a stylish design  

---

##  Technologies Used
🔹 **Python** (Backend)  
🔹 **YOLOv8** (Object Detection Model)  
🔹 **OpenCV** (Image Processing)  
🔹 **Flask** (Web Framework)  
🔹 **HTML, CSS, JavaScript** (Frontend)  

---

## Dataset & Model Training
- **Dataset Format:** Pascal VOC XML annotations converted to YOLOv8 format  
- **Model 1:** Trained YOLOv8 model for **person detection**  
- **Model 2:** Trained YOLOv8 model for **PPE detection** on cropped person images  
- **Training Steps:**
  1. Convert **Pascal VOC annotations** to YOLO format (`convert_ppe_annotations.py`)
  2. Train **Person Detection Model** (`train_person_detection.py`)
  3. Train **PPE Detection Model** (`train_ppe_detection.py`)
  4. Run inference using both models (`inference.py`)

---

##  Installation & Setup

### ** Clone the Repository**
```bash
git clone [https://github.com/your-repo/PPE-Detection.git](https://github.com/mars2812/Persons-and-PPE-Detection-using-AI)
cd PPE-Detection

### Install Dependencies
pip install -r requirements.txt

### Download YOLOv8 Model
Download a pre-trained YOLOv8 model: pip install ultralytics

### How to Run the Project
Step 1: Run Model Inference
python inference.py
This will process images and detect persons and PPE.

Step 2: Start the Flask Web App
python app.py

Note: before running inference.py file make sure that you have run pascalVOLtoYOlO file and trainning both person detection model and PPE detection model 




