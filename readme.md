# PotaKu API - API Prediksi Penyakit Kentang 🥔
<div align="center">
  <h3>API Machine Learning untuk Deteksi Penyakit pada Tanaman Kentang</h3>
</div>

## 💻 Tentang API

PotaKu API adalah layanan REST API yang menggunakan machine learning untuk mendeteksi penyakit pada tanaman kentang berdasarkan gambar daun. API ini dapat mengidentifikasi tiga kondisi: Early Blight, Late Blight, dan Healthy (sehat).

## ✨ Fitur Utama

- 🔍 **Deteksi Penyakit Kentang**: Menganalisis gambar daun kentang untuk mendeteksi penyakit
- 📊 **Tingkat Kepercayaan**: Memberikan persentase confidence dari prediksi
- 🖼️ **Format Gambar Fleksibel**: Mendukung berbagai format gambar (JPG, PNG, dll)
- ⚡ **Response Cepat**: Prediksi real-time dengan response JSON
- 🛡️ **Error Handling**: Penanganan error yang komprehensif

## 📋 Klasifikasi Penyakit

API ini dapat mendeteksi 3 kondisi pada tanaman kentang:
1. **Potato Early blight** - Penyakit bercak coklat awal
2. **Potato Late blight** - Penyakit busuk daun
3. **Potato healthy** - Kondisi sehat

## 🛠️ Teknologi yang Digunakan

- **Framework**: Flask (Python)
- **Language** : Python
- **Machine Learning**: TensorFlow/Keras
- **Image Processing**: PIL (Pillow)
- **Data Processing**: NumPy
- **Web Server**: Gunicorn
- **Deployment**: Railway

## 🔧 Model Information

- **Model Type**: Convolutional Neural Network (CNN)
- **Arsitektur**: MobileNetV2
- **Input Size**: 224x224 pixels
- **Color Mode**: RGB

## 👨‍💻 Developer

- **Nama**: [Doni Julyano Risdianto]
- **Email**: [julyanorisdianto@gmail.com]
- **GitHub**: [@donniejr07](https://github.com/donniejr07)

<div align="center">
  <p>© 2025 PotaKu API. All rights reserved.</p>
</div>
