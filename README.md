# 1. Introduction and Problem Understanding

Konteks Tujuan dari penelitian ini adalah untuk mengembangkan model klasifikasi yang kuat untuk data gambar dengan menggunakan teknik deep learning. Dataset terdiri dari 7.500 gambar secara keseluruhan, dibagi menjadi 15 kelas yang berbeda, masing-masing mewakili jenis sayuran yang berbeda.

**Konteks:**
* Load Data: Kami menggunakan ImageDataGenerator untuk memuat dan melakukan praproses terhadap 7.500 gambar secara berkelompok, untuk mengoptimalkan efisiensi komputasi.
* Augmentasi Data: Teknik seperti rotasi, pembalikan horizontal, pembesaran, geseran, dan pergeseran diterapkan untuk meningkatkan variasi dataset dan mencegah pencocokan yang berlebihan.
* Arsitektur Model: Kami menggunakan  model CNN dengan 4 lapisan konvolusi dan pooling untuk ekstraksi fitur.Flatten layer untuk meratakan hasil ekstraksi fitur.Dense layerdengan 15 neuron untuk klasifikasi ke dalam 15 kelas menggunakan probabilitas softmax.Optimizer Adam untuk pembelajaran cepat dan fungsi loss categorical cross-entropy untuk menangani multi-kelas. dan hasil akhir di dapatkan akurasi sebesar 0.8832 - loss: 0.3521 - val_accuracy: 0.9277 - val_loss: 0.2476

# 2. App Deploy
link : https://image-classifcation-use-cnn-j7nfty7afevr5dwcgacc9h.streamlit.app/
