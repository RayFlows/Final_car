# 🤖 BabyMotion: A Multifunctional Intelligent Baby Companion Robot

---

## 🌟 **Overview**

Welcome to **BabyMotion**, a revolutionary multifunctional intelligent baby companion robot developed by the collaborative efforts of **Wang Yuanhang, Liu Yijia, Rui Yuhan, and Liu Zheyi**. Designed to act as a 24/7 "childcare expert," BabyMotion combines advanced robotics, deep learning, and hardware engineering to provide comprehensive safety protection, interactive engagement, and data-driven insights for your baby.

This README provides an overview of the project, its features, technical architecture, and the team's contributions.

---

## 🔑 **Key Features**

### 🛡️ **Safety Warning System**
- Monitors the baby’s environment in real-time.
- Alerts caregivers when dangerous poses or situations are detected.

### 💬 **Intelligent Interaction Module**
- **Face Recognition Unlock**: Secure access using RetinaFace and Facenet.
- **Gesture Recognition**: Supports over 7 gestures with customizable options using Mediapipe and LSTM.
- **Voice Assistant**: Full-stack integration of Speech-to-Text, Large Language Models, and Text-to-Speech technologies.

### 🤲 **Autonomous Grasping and Classification**
- Object detection and classification using YOLO and affine transformation.
- Autonomous adjustment of the vehicle’s position for object grasping.

### 📊 **Data Management**
- SQLite database for structured storage of baby cry analysis results.
- Historical data review and trend observation for better caregiving decisions.

### 🧍 **Baby Pose Classification**
- Detects and classifies baby poses using YOLO and HRNet.
- Alerts caregivers when potentially harmful poses are identified.

### 😢 **Baby Cry Classification**
- Advanced AST model to analyze baby cry audio and determine probable reasons.

---

## 🛠️ **Technical Architecture**

### **Hardware Design**
- **Mechanical Design & Assembly**  
  Led by Wang Yuanhang, the hardware includes a mobility system, storage box, display, microphone, lights, and a robotic arm. All wiring connections were meticulously handled to ensure smooth module coordination.
  
- **Exterior Design**  
  A clean, compact, and user-friendly design balances aesthetics and functionality.

### **Software Development**
- **System Architecture**  
  Built on Raspberry Pi, the system integrates multi-device communication, control logic, and a DL-to-actuation pipeline. Full-stack development was led by Wang Yuanhang.

- **Arduino Control Coding**  
  Low-level control code written on Arduino ensures precise movement and action coordination.

- **Deep Learning Models**  
  - **Face Recognition Unlock**: RetinaFace + Facenet for secure access.  
  - **Gesture Recognition**: Mediapipe + LSTM for gesture support.  
  - **Object Detection & Classification**: YOLO for autonomous grasping.  
  - **Baby Pose Classification**: YOLO + HRNet + MLP for pose analysis.  
  - **Baby Cry Classification**: AST model for audio feature extraction.  

- **Voice Assistant**  
  Integrates Speech-to-Text, Large Language Models, and Text-to-Speech for seamless interaction.

- **Data Persistence**  
  SQLite database ensures structured storage and management of analysis results.

---

## 👥 **Team Contributions**

### **Wang Yuanhang**
- Mechanical design and baseline modeling.  
- System architecture and Raspberry Pi control logic.  
- Multi-device communication and end-to-end development.

### **Liu Zheyi**
- Hardware assembly and key module design (mobility system, robotic arm, etc.).  
- Arduino control coding for smooth module coordination.  
- Exterior design for a user-friendly appearance.

### **Rui Yuhan**
- Deep learning model development:
  - Face recognition unlock.  
  - Gesture recognition.  
  - Object detection and classification.  
  - Baby pose classification.

### **Liu Yijia**
- Baby cry classification using AST models.  
- Voice assistant integration (Speech-to-Text, LLM, Text-to-Speech).  
- Data management and persistence using SQLite.  
- System integration and application development.

---

## 💻 **Installation and Setup**

1. **Hardware Requirements**  
   - Raspberry Pi (with necessary peripherals).  
   - Arduino board for low-level control.  
   - Robotic arm, camera, microphone, and other modules as per design.

2. **Software Dependencies**  
   - Python 3.x  
   - Libraries: OpenCV, TensorFlow, PyTorch, Mediapipe, YOLO, SQLite3, etc.  
   - Frameworks: Flask/Django for backend services.

3. **Steps to Run**  
   - Clone the repository.  
   - Install dependencies using `pip install -r requirements.txt`.  
   - Configure the SQLite database.  
   - Run the main application.

---

## 📸 **Project Images**
！[Figure: Overview of the vehicle](overview.jpeg)


---

## 🙏 **Acknowledgments**

We would like to thank the National University of Singapore for their support and resources. Special thanks to our mentors and peers for their guidance and feedback.

---

## 📞 **Contact**

For any questions or feedback, please leave your message at issues part of this github repository.

---

## 📜 **License**

© Copyright National University of Singapore. All Rights Reserved.

---
