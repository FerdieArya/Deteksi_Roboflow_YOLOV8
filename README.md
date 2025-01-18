# Deteksi_Roboflow_YOLOV8
Berikut adalah kodingan untuk Algoritma Deteksi Objek Menggunakan Roboflow sebagai dataset:
```python
# Langkah 1: Instalasi dan Import Library
!pip install ultralytics roboflow
from roboflow import Roboflow
from ultralytics import YOLO
from google.colab import files
from IPython.display import Image, display
import os
import glob

# Langkah 2: Mengunduh Dataset dari Roboflow
rf = Roboflow(api_key="FZj6uhGTpecIfS65lrlc")
project = rf.workspace("pengolahan-citra-digital-hhi67").project("deteksiapple")
version = project.version(1)
dataset = version.download("yolov8")

# Langkah 3: Memuat Model yang Sudah Dilatih
model = YOLO("runs/detect/train/weights/best.pt")  # Ganti dengan path model yang sudah dilatih

#model = YOLO("yolov8n.pt")  # Gunakan YOLOv8 versi Nano untuk kecepatan
#data_path = dataset.location + "/data.yaml"  # Path file data.yaml yang disediakan oleh Roboflow

# Melatih model
#model.train(data=data_path, epochs=50, imgsz=640)

# Langkah 4: Input Gambar untuk Deteksi
print("Silakan unggah gambar yang ingin dideteksi:")
uploaded = files.upload()  # Mengunggah file gambar

# Deteksi dan simpan hasilnya
for file_name in uploaded.keys():
    print(f"Gambar berhasil diunggah: {file_name}")
    
    # Melakukan prediksi
    result = model.predict(source=file_name, save=True)
    
    # Langkah 5: Mencari folder terbaru yang berisi hasil deteksi
    result_dir_parent = "runs/detect"  # Direktori induk hasil deteksi
    result_dirs = glob.glob(os.path.join(result_dir_parent, "predict*"))  # Cari folder yang namanya diawali dengan 'predict'
    
    if result_dirs:
        # Menemukan folder terbaru berdasarkan waktu pembuatan
        latest_result_dir = max(result_dirs, key=os.path.getmtime)  # Menentukan folder yang paling baru
        print(f"Folder hasil deteksi terbaru: {latest_result_dir}")
        
        # Menemukan file gambar hasil deteksi dalam folder terbaru
        detected_files = glob.glob(os.path.join(latest_result_dir, "*"))  # Mengambil semua file di dalam folder
        detected_image_files = [f for f in detected_files if f.endswith(('.jpg', '.jpeg', '.png'))]  # Filter hanya gambar
        
        if detected_image_files:
            output_image_path = detected_image_files[0]  # Ambil gambar pertama yang ditemukan
            print(f"Hasil deteksi disimpan di: {output_image_path}")
            
            # Tampilkan gambar hasil deteksi
            display(Image(filename=output_image_path))
        else:
            print(f"Tidak ada gambar hasil deteksi di folder {latest_result_dir}.")
    else:
        print(f"Tidak ada folder hasil deteksi ditemukan di {result_dir_parent}.")

```
Hasil Output: 

![HasilDeteksi](https://github.com/user-attachments/assets/f8df2b88-c591-41f2-903e-7445f01e4fab)
