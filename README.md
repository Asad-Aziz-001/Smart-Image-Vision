<div align="center">

# 🤖 Smart Image Vision

**AI-Powered Image Analysis with BLIP + Dominant Color Detection**  
Instantly understand any photo using state-of-the-art vision-language models.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

</div>

## ✨ Features

- **Natural Language Description** using **Salesforce BLIP** (top-tier image captioning model)
- **Dominant Color Detection** with RGB values & color name
- Stunning **2025-level glassmorphic UI** with gradient backgrounds and smooth animations
- Zero setup – just upload and get results in seconds
- Fully responsive & mobile-friendly

## 🚀 Live Demo

https://smart-image-vision.streamlit.app  
*(Deployed instantly via Streamlit Community Cloud)*

## 🛠 Tech Stack

| Technology              | Purpose                          |
|-------------------------|----------------------------------|
| Streamlit               | Beautiful web interface          |
| Hugging Face Transformers | BLIP model for captioning       |
| PyTorch                 | Deep learning backend            |
| PIL + scikit-learn      | Dominant color extraction (KMeans)|
| Custom CSS              | Modern gradient & glassmorphic design |

## 📦 Quick Start (Local)

```bash
git clone https://github.com/yourusername/smart-image-vision.git
cd smart-image-vision

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📄 Create `requirements.txt`

```txt
streamlit
torch
transformers
pillow
scikit-learn
numpy
```

Run this command to generate it automatically:
```bash
pip freeze > requirements.txt
```

## 🌐 Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. New app → Select your repo → `app.py` → Deploy!

## 🎨 Preview

![Screenshot 1](https://raw.githubusercontent.com/yourusername/smart-image-vision/main/screenshot1.png)  
![Screenshot 2](https://raw.githubusercontent.com/yourusername/smart-image-vision/main/screenshot2.png)

> Pro tip: Take 2–3 clean screenshots of your app and upload them to the repo!

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

## 👨‍💻 Author

**Asad Aziz**  
- GitHub: [@yourusername](https://github.com/yourusername)  
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

## 📜 License

MIT License – feel free to use, modify, and share!

---

<div align="center">
  <strong>Built with passion in 2025</strong> <br/>
  ⭐ Star this repo if you found it useful!
</div>
```
