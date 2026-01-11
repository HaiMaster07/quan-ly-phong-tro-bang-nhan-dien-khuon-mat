# Hệ thống quản lý phòng trọ bằng nhận diện khuôn mặt

Ứng dụng hỗ trợ quản lý phòng trọ, khách thuê và nhận diện khuôn mặt thông qua camera nhằm phát hiện người thuê và người lạ ra vào khu vực phòng trọ.

## 1. Tính năng chính
* Nhận diện khuôn mặt theo thời gian thực từ camera
* Phân biệt các khách thuê với nhau và người lạ
  
  ![Ảnh chụp màn hình 2026-01-11 110708](https://github.com/HaiMaster07/quan-ly-phong-tro-bang-nhan-dien-khuon-mat/blob/master/photo/%E1%BA%A2nh%20ch%E1%BB%A5p%20m%C3%A0n%20h%C3%ACnh%202026-01-11%20110708.png)
* Quản lý phòng trọ (thêm, xóa, cập nhật)

  ![Ảnh chụp màn hình 2026-01-11 111210](https://github.com/HaiMaster07/quan-ly-phong-tro-bang-nhan-dien-khuon-mat/blob/master/photo/%E1%BA%A2nh%20ch%E1%BB%A5p%20m%C3%A0n%20h%C3%ACnh%202026-01-11%20111210.png)
* Quản lý thông tin khách thuê
  
  ![Ảnh chụp màn hình 2026-01-11 111129](https://github.com/HaiMaster07/quan-ly-phong-tro-bang-nhan-dien-khuon-mat/blob/master/photo/%E1%BA%A2nh%20ch%E1%BB%A5p%20m%C3%A0n%20h%C3%ACnh%202026-01-11%20111129.png)
* Tự động đồng bộ trạng thái phòng theo số lượng khách
* Lưu trữ dữ liệu bằng file Excel

![Ảnh chụp màn hình 2026-01-11 121825](https://github.com/HaiMaster07/quan-ly-phong-tro-bang-nhan-dien-khuon-mat/blob/master/photo/%E1%BA%A2nh%20ch%E1%BB%A5p%20m%C3%A0n%20h%C3%ACnh%202026-01-11%20121825.png)

![Ảnh chụp màn hình 2026-01-11 121906](https://github.com/HaiMaster07/quan-ly-phong-tro-bang-nhan-dien-khuon-mat/blob/master/photo/%E1%BA%A2nh%20ch%E1%BB%A5p%20m%C3%A0n%20h%C3%ACnh%202026-01-11%20121906.png)

## 2. Cài đặt
  - Sao chép code dự án: [Tại đây](https://github.com/HaiMaster07/quan-ly-phong-tro-bang-nhan-dien-khuon-mat/blob/master/hostel_app.py)
  - Cài đặt thư viện cần thiết
Bạn cần cài đặt một số thư viện Python. Sử dụng các lệnh sau để cài đặt chúng:
pip install opencv-python
pip install numpy
pip install pandas
pip install openpyxl
pip install pillow
pip install click
pip install tkcalendar
  - Face_recogntion. Để cài cần làm theo các bước trong file [này](https://github.com/HaiMaster07/quan-ly-phong-tro-bang-nhan-dien-khuon-mat/tree/master/instruct)
## 3. Cấu trúc thư mục

```
hostel_app/
│
├── data/
│   ├── face_data/
│   │   ├── User.111/
│   │   │   ├── User.111.1.jpg
│   │   │   ├── User.111.2.jpg
│   │   │   ├── User.111.3.jpg
│   │   │   └── ...
│   │
│   ├── face_encodings.pkl
│   ├── customers.xlsx
│   └── rooms.xlsx
│
└── hostel_app.py

```
## 4. Sử dụng
### Khởi động chương trình

Sau khi cài đặt đầy đủ các thư viện cần thiết, chạy file chính của hệ thống: python main.py

Giao diện quản lý khu phòng trọ sẽ được hiển thị.

### Quản lý phòng trọ

* Nhập mã phòng, số người tối đa

* Nhấn nút Thêm phòng để lưu dữ liệu

* Danh sách phòng được cập nhật ngay trên giao diện

* Mỗi phòng được định danh duy nhất bằng mã phòng.
### Quản lí khách thuê

* Nhập thông tin khách thuê

* Chọn phòng tương ứng

* Thực hiện chụp ảnh khuôn mặt để lưu dữ liệu nhận diện

* Hệ thống tự động trích xuất và lưu vector đặc trưng khuôn mặt.
### Nhận diện khuôn mặt

* Nhấn nút Bật camera

* Camera bắt đầu thu nhận hình ảnh

* Hệ thống tự động:

* Phát hiện khuôn mặt

* So sánh với dữ liệu đã lưu

* Hiển thị kết quả nhận diện (người thuê(tên, mã khách) / người lạ)
### Đồng bộ dữ liệu
* Sau mỗi thao tác thêm, xóa hoặc cập nhật khách thuê, hệ thống:

* Tự động cập nhật số lượng khách trong từng phòng

* Cập nhật trạng thái phòng (trống / đã thuê)

* Không yêu cầu thao tác thủ công từ người dùng
### Lưu ý khi sử dụng
* Nên sử dụng trong điều kiện ánh sáng tốt

* Tránh che khuôn mặt hoặc quay mặt quá lệch

* Camera cần được kết nối và hoạt động ổn định
## 5. Cơ chế nhận diện khuôn mặt
Khi thêm người dùng mới, hệ thống sử dụng camera chụp liên tục ~50 ảnh khuôn mặt ở nhiều góc độ khác nhau.

Các ảnh được lưu vào thư mục data/face_data/User.xxx/ để đảm bảo độ đa dạng (nghiêng trái/phải, thay đổi biểu cảm).

Mỗi ảnh khuôn mặt được trích xuất vector đặc trưng (face encoding) bằng thư viện face_recognition.

Tất cả vector được tổng hợp và lưu lại vào file face_encodings.pkl để sử dụng cho các lần nhận diện sau.

Khi nhận diện, hệ thống so sánh vector khuôn mặt từ camera với dữ liệu đã lưu bằng khoảng cách Euclidean.

Nếu khoảng cách nhỏ hơn ngưỡng cho phép → xác định đúng người dùng, ngược lại được xem là Người lạ.
