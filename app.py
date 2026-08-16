import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Streamlit Page Config
st.set_page_config(page_title="Waste Classification App", page_icon="♻️", layout="centered")

# Custom CSS for Full Screen Glassy White Blurred Background
page_bg_css = """
<style>
/* 1. Put the abstract fluid image on the root app background */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1550684848-fac1c5b4e853?q=80&w=2564&auto=format&fit=crop") !important;
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* 2. Blur the ENTIRE screen and give it a glassy white tint */
[data-testid="stAppViewContainer"] {
    background-color: rgba(255, 255, 255, 0.3) !important;
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
}
[data-testid="stHeader"] {
    background: transparent !important;
}

/* Apple Crystal Glassmorphism effect for the main container */
.apple-glass {
    background: rgba(255, 255, 255, 0.2) !important;
    border-radius: 28px;
    padding: 35px;
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.6);
    text-align: center;
    margin-bottom: 30px;
    margin-top: 20px;
}
.apple-glass h1 {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #ffffff !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.4);
    font-weight: 800;
    margin: 0;
    padding: 0;
    letter-spacing: -0.5px;
}
.apple-glass p {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #ffffff !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.4);
    font-size: 1.15rem;
    margin-top: 15px;
    font-weight: 500;
}

/* Global Font and Text Color (White with deep shadow for perfect legibility) */
h1, h2, h3, h4, h5, h6, p, span, div, label, li {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #ffffff !important;
    text-shadow: 0 1px 5px rgba(0,0,0,0.4);
}

/* Glass Button */
.stButton>button {
    background: rgba(255, 255, 255, 0.2) !important;
    color: #ffffff !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
}
.stButton>button:hover {
    background: rgba(255, 255, 255, 0.4) !important;
    transform: translateY(-2px) scale(1.02);
}
.stButton>button p {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 17px !important;
}

/* Completely fix the Uploader Box styling */
[data-testid="stFileUploadDropzone"] {
    background-color: rgba(255, 255, 255, 0.2) !important;
    border: 2px dashed rgba(255, 255, 255, 0.9) !important;
    border-radius: 20px !important;
    padding: 20px !important;
}
[data-testid="stFileUploadDropzone"] div,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small {
    color: #ffffff !important;
    text-shadow: 0px 2px 4px rgba(0,0,0,0.8) !important; /* Heavy shadow so it NEVER blends in */
}
[data-testid="stFileUploadDropzone"] svg {
    fill: #ffffff !important;
    filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.8)) !important;
}
/* Specifically the upload button inside the uploader */
[data-testid="stFileUploadDropzone"] button {
    background-color: rgba(0, 0, 0, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    text-shadow: none !important;
}
[data-testid="stFileUploadDropzone"] button:hover {
    background-color: rgba(0, 0, 0, 0.8) !important;
}
[data-testid="stFileUploadDropzone"] button p {
    text-shadow: none !important;
}

/* Camera Input styling */
[data-testid="stCameraInput"] {
    border-radius: 20px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
}

/* Tabs styling */
[data-testid="stTabs"] button {
    background-color: transparent !important;
    border: none !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: rgba(255,255,255,0.3) !important;
    border-radius: 12px;
}

/* Success and Error messages */
[data-testid="stAlert"] {
    background-color: rgba(255, 255, 255, 0.2) !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    border-radius: 18px !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
}
</style>
"""

# Inject CSS
st.markdown(page_bg_css, unsafe_allow_html=True)

# Define Model Architecture
class wasteCNN(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.network = nn.Sequential (
            nn.Conv2d(3,32 , kernel_size=(3,3), stride=1 , padding="same"), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), 

            nn.Conv2d(32,64 , kernel_size=(3,3), stride=1 , padding="same"),  
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), 
            
            nn.Conv2d(64,128 , kernel_size=(3,3), stride=1 , padding="same"),  
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), 

            nn.Conv2d(128,256 , kernel_size=(3,3), stride=1 , padding="same"),  
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), 

            nn.Conv2d(256,512 , kernel_size=(3,3), stride=1 , padding="same"),  
            nn.BatchNorm2d(512), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), 
        )

        self.fc_layers = nn.Sequential (
            nn.Flatten(), 
            nn.Linear(512*7*7 , 600), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(600,120), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(120,1), 
            nn.Sigmoid()
        )

    def forward(self,x): 
        x = self.network(x)
        x = self.fc_layers(x)
        return x

@st.cache_resource
def load_model():
    model = wasteCNN()
    model.load_state_dict(torch.load('best_waste_classifier.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Transforms from the notebook
test_transforms = transforms.Compose ([
    transforms.Resize((224,224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apple Glass Header
st.markdown('''
<div class="apple-glass">
    <h1>♻️ Waste Classification</h1>
    <p>Upload an image to instantly classify waste as Organic or Non-organic.</p>
</div>
''', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Determine which image source to use
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

if image is not None:
    # Show uploaded image
    st.image(image, caption='Image to Classify', use_container_width=True)
    
    st.write("")
    if st.button("Classify Image", use_container_width=True):
        with st.spinner('Analyzing image...'):
            input_tensor = test_transforms(image).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                prediction = output.item()
            
            st.markdown("---")
            if prediction < 0.5:
                st.success(f"### 🌱 **Prediction: Organic Waste**\n\n**Confidence:** {(1 - prediction)*100:.2f}%")
            else:
                st.warning(f"### ♻️ **Prediction: Non-organic / Recyclable Waste**\n\n**Confidence:** {prediction*100:.2f}%")
