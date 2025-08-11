import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import cv2
import random
import pyttsx3
from googletrans import Translator
from gtts import gTTS
import os
import tempfile
import playsound
import open3d as o3d 
import numpy as np



class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Flatten(),
            torch.nn.Linear(16384, 128),
            torch.nn.Linear(128, 16),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.sequential(x)

model_path = "G:/My Drive/fracture_detetction_app/models/boneFrac.pt"
model = ConvNet()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


fracture_stories = [
    "It looks like your bone got a little crack, kind of like when glass chips but doesn’t break all the way.",
    "There’s a tiny break, like when a wooden stick bends and snaps a bit. Don’t worry, it will heal with care.",
    "Imagine a phone screen with a small crack—this fracture is like that. Some rest will fix it up.",
    "Looks like your bone had a rough moment, like a cookie that snapped. Time and care will make it better.",
    "Think of a toy that took a tumble and cracked a little—it’s fixable, and so is this!",
    "This is like a bend in a paperclip—it’s a small injury, but with the right treatment, it can be straightened out.",
    "Your bone has taken a hit like a dropped ceramic mug—it might be cracked, but it’s not beyond repair.",
    "Imagine a branch on a tree that bends too far in the wind—that’s what your bone is going through. It'll bounce back with care.",
    "This is like a favorite book with a torn page—it doesn’t ruin the story, and it can always be fixed.",
    "It seems like a speed bump in your healing journey. With a bit of rest and the right help, you’ll be back on track.",
    "Your bone has a battle scar—proof that it’s been through something. Healing is just around the corner.",
    "Like a dented bicycle frame, it might feel off now, but repairs can bring it back to full strength.",
    "This is your body’s way of asking for a little break. Rest, recovery, and you’ll be good to go.",
    "Fractures are like thunderstorms—they seem rough, but they always pass. Sunshine (and healing) is coming soon.",
    "A small crack doesn’t define the whole structure—your body is still strong and capable of bouncing back!"
]

languages = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn"
}

translator = Translator()

def get_random_fracture_story():
    return random.choice(fracture_stories)

def speak_text(text, lang_code):
    root.update_idletasks()

    if lang_code == "en":
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            messagebox.showerror("Voice Error", f"Voice-over failed: {str(e)}")
    else:
        try:
            tts = gTTS(text=text, lang=lang_code)
            with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
                tts.save(fp.name)
                playsound.playsound(fp.name)
        except Exception as e:
            messagebox.showerror("Voice Error", f"Voice-over failed: {str(e)}")

    root.update_idletasks()

def translate_text(text, lang_code):
    if lang_code == "en":
        return text
    try:
        translated = translator.translate(text, dest=lang_code)
        return translated.text
    except Exception as e:
        return f"Translation error: {str(e)}"

def predict_image(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()
    label = "Fractured" if prediction > 0.5 else "Not Fractured"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence


def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("X-ray Images", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

  
    show_3d_visualization.last_image_path = file_path


    img_cv = cv2.imread(file_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv)
    img_pil.thumbnail((250, 250))
    img_tk = ImageTk.PhotoImage(img_pil)

    image_label.config(image=img_tk)
    image_label.image = img_tk

    result, confidence = predict_image(file_path)
    lang_code = languages[lang_choice.get()]

    confidence_percent = f"{confidence * 100:.2f}%"
    result_string = f"Prediction: {result} (Confidence: {confidence_percent})"

    if result == "Fractured":
        story = get_random_fracture_story() + "\n\n👉 Please visit a doctor for proper examination."
    else:
        story = "✅ The bone looks healthy and strong!"

    translated_result = translate_text(result_string, lang_code)
    translated_story = translate_text(story, lang_code)

    result_text.set(translated_result)
    story_text.set(translated_story)
    is_fractured.set(result == "Fractured") 

    if messagebox.askyesno("Voice Over", "Would you like to hear the results?"):
        speak_text(translated_result + ". " + translated_story, lang_code)


def toggle_theme(*args):
    if theme.get() == "Light":
        root.configure(bg="#f0f4f8")
        style.configure("TLabel", background="#f0f4f8", foreground="#000")
        style.configure("Title.TLabel", background="#f0f4f8", foreground="#000")
    else:
        root.configure(bg="#2e2e2e")
        style.configure("TLabel", background="#2e2e2e", foreground="#ffffff")
        style.configure("Title.TLabel", background="#2e2e2e", foreground="#ffffff")


def show_3d_visualization():
    try:
        if not hasattr(show_3d_visualization, "last_image_path") or not show_3d_visualization.last_image_path:
            messagebox.showwarning("No Image", "Please upload an X-ray image first.")
            return

        img = cv2.imread(show_3d_visualization.last_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not load the image for 3D visualization.")
        img = cv2.resize(img, (256, 256))

        h, w = img.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        z = img.astype(np.float32) / 255.0  # normalize intensities

        points = np.stack((x, y, z * 50), axis=-1).reshape(-1, 3)
        colors = np.repeat(img.reshape(-1, 1) / 255.0, 3, axis=1)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pc], window_name="🦴 3D X-ray Visualization")

    except Exception as e:
        messagebox.showerror("3D Error", f"Unable to display 3D bone: {str(e)}")



root = tk.Tk()
root.title("🦴 Bone Fracture Detector")
root.geometry("520x680")
root.configure(bg="#f0f4f8")

style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 12), padding=6)
style.configure("TLabel", font=("Segoe UI", 12), background="#f0f4f8")
style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))


ttk.Label(root, text="🦴 Bone Fracture Detection", style="Title.TLabel").pack(pady=20)


frame = ttk.Frame(root)
frame.pack(pady=5)

theme = tk.StringVar(value="Light")
ttk.Label(frame, text="Theme:").grid(row=0, column=0, padx=5)
ttk.OptionMenu(frame, theme, "Light", "Light", "Dark", command=toggle_theme).grid(row=0, column=1)

lang_choice = tk.StringVar(value="English")
ttk.Label(frame, text="Language:").grid(row=0, column=2, padx=5)
ttk.OptionMenu(frame, lang_choice, "English", *languages.keys()).grid(row=0, column=3)


image_label = ttk.Label(root)
image_label.pack(pady=10)


ttk.Button(root, text="📤 Upload X-ray Image", command=load_image).pack(pady=15)
ttk.Button(root, text="🦴 Show 3D Bone", command=show_3d_visualization).pack(pady=5)


result_text = tk.StringVar()
story_text = tk.StringVar()
is_fractured = tk.BooleanVar(value=False) 

ttk.Label(root, textvariable=result_text, font=("Segoe UI", 14, "bold"), foreground="#1a73e8").pack(pady=10)
ttk.Label(root, textvariable=story_text, wraplength=470, justify="center", font=("Segoe UI", 11), foreground="#444").pack(pady=15)


ttk.Button(root, text="❌ Exit", command=root.quit).pack(pady=10)



ttk.Label(root, text="Created by The Think Tank", font=("Segoe UI", 9, "italic"), foreground="#888").pack(side="bottom", pady=10)


root.mainloop()
