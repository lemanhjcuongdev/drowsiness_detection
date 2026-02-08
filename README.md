# Hướng Dẫn Cài Đặt & Chạy - Hệ Thống Phát Hiện Sự Chú Ý

## 📋 Yêu Cầu Hệ Thống

| Thành phần | Phiên bản | Ghi chú |
|------------|-----------|---------|
| **Python** | 3.6.x | Khuyến nghị 3.6.8 (dlib tương thích tốt nhất) |
| **OS** | Windows 10/11 | Đã test trên Windows |
| **Webcam** | Bất kỳ | Webcam laptop hoặc USB |
| **RAM** | ≥ 4GB | Khuyến nghị 8GB |

---

## 🔧 Hướng Dẫn Cài Đặt

### Bước 1: Cài đặt Python 3.6.8

1. Tải Python 3.6.8 từ: https://www.python.org/downloads/release/python-368/
2. Chọn **Windows x86-64 executable installer**
3. **QUAN TRỌNG**: Tick ✅ "Add Python to PATH" khi cài đặt

### Bước 2: Tạo Virtual Environment

```powershell
cd d:\CUONG\PTIT\THI_GIAC_MAY_TINH\drowsiness_detection

# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
.venv\Scripts\activate
```

### Bước 3: Cài đặt thư viện

```powershell
# Cài các thư viện cơ bản trước
pip install numpy==1.16.2
pip install scipy==1.2.1
pip install opencv-contrib-python==4.0.0.21
pip install imutils==0.5.2

# Cài dlib (quan trọng - cần cài từ file wheel)
pip install dlib-19.17.0-cp36-cp36m-win_amd64.whl

# Cài face_recognition
pip install face_recognition==1.2.3
```

> ⚠️ **Lưu ý về dlib**: File `.whl` đã có sẵn trong thư mục project. Nếu gặp lỗi, cần cài CMake và Visual Studio Build Tools.

### Bước 4: Tải model landmarks (nếu chưa có)

File `shape_predictor_68_face_landmarks.dat` (~100MB) đã có sẵn trong project.

Nếu cần tải lại: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

---

## 🚀 Chạy Chương Trình

```powershell
# Kích hoạt môi trường ảo (nếu chưa)
.venv\Scripts\activate

# Chạy chương trình
python sleep_detect.py
```

### Phím tắt khi chạy:
- **`C`** - Calibrate góc nhìn thẳng (nhìn thẳng vào webcam rồi bấm)
- **`Q`** - Thoát chương trình

---

## ⚠️ Các Lỗi Thường Gặp & Cách Khắc Phục

### Lỗi 1: `No module named 'dlib'`
**Nguyên nhân**: dlib chưa cài hoặc cài sai version

**Cách sửa**:
```powershell
pip install dlib-19.17.0-cp36-cp36m-win_amd64.whl
```

### Lỗi 2: `Could not find shape_predictor_68_face_landmarks.dat`
**Nguyên nhân**: Thiếu file model

**Cách sửa**: Đảm bảo file `shape_predictor_68_face_landmarks.dat` nằm cùng thư mục với `sleep_detect.py`

### Lỗi 3: `cv2.error: (-215:Assertion failed)` hoặc webcam không mở
**Nguyên nhân**: Webcam đang được sử dụng bởi ứng dụng khác

**Cách sửa**: 
- Tắt các ứng dụng khác đang dùng webcam (Zoom, Teams, ...)
- Thử đổi `cv2.VideoCapture(0)` thành `cv2.VideoCapture(1)` nếu có nhiều camera

### Lỗi 4: Python version không đúng
**Nguyên nhân**: Dùng Python 3.7+ với file dlib cho 3.6

**Cách sửa**: Dùng Python 3.6.8 hoặc tìm file dlib wheel phù hợp version

---

## 📁 Cấu Trúc Thư Mục

```
drowsiness_detection/
├── sleep_detect.py              # File chính
├── shape_predictor_68_face_landmarks.dat  # Model dlib (100MB)
├── dlib-19.17.0-cp36-cp36m-win_amd64.whl  # Wheel cài dlib
├── setup.txt                    # Danh sách thư viện (cũ)
├── .venv/                       # Môi trường ảo Python
└── README.md                    # File hướng dẫn này
```

---

## 📝 Thư Viện Sử Dụng (Phiên Bản Mới)

```
numpy==1.16.2
scipy==1.2.1
opencv-contrib-python==4.0.0.21
imutils==0.5.2
dlib==19.17.0
face_recognition==1.2.3
```

> ℹ️ **Ghi chú**: Code đã được cập nhật để **không cần** `keras` và `tensorflow` nữa. Phát hiện mắt giờ dùng thuật toán EAR thay vì deep learning.
