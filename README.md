## Ai-Xray-Analyzer-using-Convolutional-Neural-Networks-CNN
This project presents an AI-powered solution for detecting bone fractures from X-ray images using deep learning. By leveraging Convolutional Neural Networks (CNNs), the system classifies uploaded X-ray images as either fractured or normal, providing quick and accurate assistance to radiologists and patients.
A 3D visualisation of the fractured bone enhances user understanding of the diagnosis.

---

## Features


* Upload and predict fractures from X-ray images
* Confidence score display for each prediction
* Multilingual voice output (English, Hindi, Tamil, Kannada)
* Storytelling-style explanation for fractured cases
* 3D X-ray visualization using Open3D
* Light/Dark theme switch for GUI

---


# Technologies Used
* Python
* PyTorch (for CNN model)
* Tkinter (GUI)
* OpenCV (image handling)
* gTTS & pyttsx3 (text-to-speech)
* Open3D (3D visualization)
* Googletrans (translation)
  git clone https://github.com/your-username/Ai-Xray-Analyzer-using-Convolutional-Neural-Networks-CNN-.git
  cd bone-fracture-detector

---

### 🚀 How to Run

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/bone-fracture-detector.git
cd bone-fracture-detector
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Download or Place the Model Weights**
   Ensure your trained PyTorch model (`boneFrac.pt`) is placed in the correct path and update it in the code:

```python
model = torch.load('path/to/boneFrac.pt')
```

4. **Run the App**

```bash
python app.py
```

---

### 👤 Author

**Saniha Manjunath**
GitHub: [@saniha139](https://github.com/saniha139)

---

