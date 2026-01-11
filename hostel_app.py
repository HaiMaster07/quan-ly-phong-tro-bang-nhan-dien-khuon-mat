import os
import cv2
import numpy as np
import pandas as pd
import threading
import queue
import time
import traceback
import pickle
import face_recognition
from tkinter import *
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from PIL import Image, ImageTk, ImageDraw, ImageFont
from pathlib import Path
from collections import deque, defaultdict

# ============ CẤU HÌNH ============
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACE_DIR = DATA_DIR / "face_data"
CUSTOMER_FILE = DATA_DIR / "customers.xlsx"
ROOM_FILE = DATA_DIR / "rooms.xlsx"

MAX_IMAGES_PER_USER = 50
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480
FACE_DISTANCE_THRESHOLD = 0.38
TEMPORAL_WINDOW = 7
TEMPORAL_REQUIRED = 4

DATA_DIR.mkdir(parents=True, exist_ok=True)
FACE_DIR.mkdir(parents=True, exist_ok=True)

def find_system_font():
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\ARIALUNI.TTF",
        r"C:\Windows\Fonts\DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf" ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

FONT_PATH = find_system_font()

def load_font(size=16):
    try:
        if FONT_PATH:
            return ImageFont.truetype(FONT_PATH, size=size)
    except Exception:
        pass
    return ImageFont.load_default()

# ============ DATABASE ============
def ensure_excel(path, columns):
    path = Path(path)
    if not path.exists():
        df = pd.DataFrame(columns=columns)
        df.to_excel(path, index=False, engine="openpyxl")

ensure_excel(CUSTOMER_FILE, ["customer_id", "name", "dob", "phone", "start_date", "room_id"])
ensure_excel(ROOM_FILE, ["room_id", "area", "max_people", "status"])

class ExcelDB:
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        try:
            if self.path.exists():
                return pd.read_excel(self.path, engine="openpyxl", dtype=str)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def write(self, df):
        try:
            df.to_excel(self.path, index=False, engine="openpyxl")
            return True
        except Exception:
            return False

    def append(self, row):
        df = self.read()
        new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        return self.write(new_df)

    def update(self, key_col, key_val, updates):
        df = self.read()
        if df.empty or key_col not in df.columns:
            return False
        mask = df[key_col].astype(str) == str(key_val)
        if not mask.any():
            return False
        for k, v in updates.items():
            if k in df.columns:
                df.loc[mask, k] = v
        return self.write(df)

    def delete(self, key_col, key_val):
        df = self.read()
        if df.empty or key_col not in df.columns:
            return False
        df = df[df[key_col].astype(str) != str(key_val)]
        return self.write(df)

    def count_room_members(self, room_id):
        df = self.read()
        if df is None or df.empty:
            return 0
        return len(df[df['room_id'].astype(str) == str(room_id)])

# ============ FACE MANAGER ============
class FaceManager:
    def __init__(self, face_dir=str(FACE_DIR)):
        self.face_dir = Path(face_dir)
        self.encodings_file = self.face_dir / "face_encodings.pkl"
        self.known_encodings = []
        self.known_ids = []
        self.model_loaded = False

    def _user_folder(self, user_id):
        return self.face_dir / f"User.{user_id}"

    def capture_images(self, user_id, num_samples=MAX_IMAGES_PER_USER, preview_callback=None, 
                      stop_event=None, progress_callback=None):
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise RuntimeError("Cannot open camera")
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        time.sleep(1)        # Đợi camera khởi động
        folder = self._user_folder(user_id)
        folder.mkdir(parents=True, exist_ok=True)
        saved = 0
        base_count = len(list(folder.glob("*.jpg")))
        frame_skip = 0
        try:
            while saved < num_samples:
                if stop_event and stop_event.is_set():
                    break
                ret, frame = cam.read()
                if not ret or frame is None:
                    continue
                frame_skip += 1
                if frame_skip % 3 != 0:  # Chỉ xử lý mỗi 3 frame
                    if preview_callback:
                        display_frame = frame.copy()
                        info_text = f"Scanned: {saved}/{num_samples}"
                        cv2.putText(display_frame, info_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        preview_callback(display_frame)
                    continue
                # Chuyển sang RGB để xử lý với face_recognition
                if frame is None or not isinstance(frame, np.ndarray): 
                    continue
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3: 
                    continue                
                # Phát hiện khuôn mặt
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")                
                # Vẽ thông tin
                display_frame = frame.copy()
                info_text = f"Scanned: {saved}/{num_samples}"
                cv2.putText(display_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)               
                if len(face_locations) > 0:
                    top, right, bottom, left = face_locations[0]
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)                   
                    # Lưu ảnh
                    face_img_rgb = rgb_frame[top:bottom, left:right]
                    if face_img_rgb.size > 0 and face_img_rgb.ndim == 3 and face_img_rgb.shape[2] == 3:
                        base_count += 1
                        saved += 1
                        fname = folder / f"User.{user_id}.{base_count}.jpg"
                        face_img_bgr = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(fname), face_img_bgr)                    
                        if progress_callback:
                            progress_callback(saved, num_samples)               
                if preview_callback:
                    preview_callback(display_frame)                    
                time.sleep(0.1)                
        finally:
            cam.release()       
        # Tự động tạo encodings
        self.incremental_update(user_id)
        return len(list(folder.glob("*.jpg")))

    def train_model(self):
        encodings = []
        ids = []        
        for user_folder in os.listdir(self.face_dir):
            if not user_folder.startswith("User."):
                continue
            try:
                uid = int(user_folder.split(".")[1])
            except Exception:
                continue          
            folder_path = os.path.join(self.face_dir, user_folder)
            for img_name in os.listdir(folder_path):
                if not img_name.endswith(".jpg"):
                    continue
                img_path = os.path.join(folder_path, img_name)                
                img = cv2.imread(img_path)
                if img is None:
                    continue               
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb_img, model="hog")                
                if len(boxes) > 0:
                    encoding = face_recognition.face_encodings(rgb_img, boxes)[0]
                    encodings.append(encoding)
                    ids.append(uid)
        
        if not encodings:
            raise ValueError("No training data. Capture faces first.")
        
        self.known_encodings = encodings
        self.known_ids = ids       
        with open(str(self.encodings_file), 'wb') as f:
            pickle.dump({'encodings': encodings, 'ids': ids}, f)       
        self.model_loaded = True
        return len(encodings)

    def load_model(self):
        if self.model_loaded:
            return True
        if self.encodings_file.exists():
            try:
                with open(str(self.encodings_file), 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data['encodings']
                    self.known_ids = data['ids']
                self.model_loaded = True
                return True
            except Exception:
                return False
        return False

    def incremental_update(self, user_id):
        folder = self._user_folder(user_id)
        if not folder.exists():
            return 0        
        new_encodings = []
        new_ids = []        
        for img_path in folder.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue           
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_img, model="hog")            
            if len(boxes) > 0:
                encoding = face_recognition.face_encodings(rgb_img, boxes)[0]
                new_encodings.append(encoding)
                new_ids.append(int(user_id))        
        if not new_encodings:
            return 0        
        if self.encodings_file.exists():
            try:
                with open(str(self.encodings_file), 'rb') as f:
                    data = pickle.load(f)
                    old_encodings = data['encodings']
                    old_ids = data['ids']               
                filtered_encodings = []
                filtered_ids = []
                for enc, uid in zip(old_encodings, old_ids):
                    if uid != int(user_id):
                        filtered_encodings.append(enc)
                        filtered_ids.append(uid)               
                self.known_encodings = filtered_encodings + new_encodings
                self.known_ids = filtered_ids + new_ids
            except Exception:
                self.known_encodings = new_encodings
                self.known_ids = new_ids
        else:
            self.known_encodings = new_encodings
            self.known_ids = new_ids      
        with open(str(self.encodings_file), 'wb') as f:
            pickle.dump({'encodings': self.known_encodings, 'ids': self.known_ids}, f)        
        self.model_loaded = True
        return len(new_encodings)

    def recognize_face(self, face_encoding):
        if not self.model_loaded or len(self.known_encodings) == 0:
            return None, 1.0      
        distances = face_recognition.face_distance(self.known_encodings, face_encoding)       
        if len(distances) == 0:
            return None, 1.0        
        min_distance = np.min(distances)
        min_index = np.argmin(distances)        
        if min_distance < FACE_DISTANCE_THRESHOLD:
            return self.known_ids[min_index], min_distance       
        return None, min_distance

# ============ CAMERA THREAD ============
class CameraThread(threading.Thread):
    def __init__(self, out_queue, stop_event, face_mgr, customer_db):
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.face_mgr = face_mgr
        self.customer_db = customer_db
        self.font_cache = load_font(18)
        self.recent = defaultdict(lambda: deque(maxlen=TEMPORAL_WINDOW))
        self.last_display = {}
        self.frame_counter = 0

    def _bbox_key(self, left, top, right, bottom):
        qx = int(left / 20) * 20
        qy = int(top / 20) * 20
        qw = int((right - left) / 20) * 20
        qh = int((bottom - top) / 20) * 20
        return f"{qx}_{qy}_{qw}_{qh}"

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.out_queue.put(("error", "Cannot open camera"))
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        time.sleep(1)  # Đợi camera khởi động
        customers_cache = {}
        last_reload = 0
        frame_skip = 0
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                now = time.time()
                # Reload dữ liệu khách mỗi 3 giây
                if now - last_reload > 3:
                    df = self.customer_db.read()
                    customers_cache = {}
                    if not df.empty:
                        for _, r in df.iterrows():
                            cid = str(r.get("customer_id", ""))
                            name = str(r.get("name", ""))
                            customers_cache[cid] = name
                    last_reload = now
                frame_skip += 1
                if frame_skip % 5 != 0:  # chỉ xử lý mỗi 5 frame
                    continue
                # Resize nhỏ để tăng tốc
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                # Nhận diện khuôn mặt trên ảnh nhỏ
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                # Scale lại tọa độ về ảnh gốc
                scaled_locations = [(top*2, right*2, bottom*2, left*2) for (top, right, bottom, left) in face_locations]
                display = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
                scale_x = PREVIEW_WIDTH / frame.shape[1]
                scale_y = PREVIEW_HEIGHT / frame.shape[0]

                for (top, right, bottom, left), face_encoding in zip(scaled_locations, face_encodings):
                    dx1 = int(left * scale_x)
                    dy1 = int(top * scale_y)
                    dx2 = int(right * scale_x)
                    dy2 = int(bottom * scale_y)

                    cv2.rectangle(display, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)

                    label_text = "Người Lạ"
                    color_rgb = (255, 0, 0)

                    if self.face_mgr.load_model():
                        try:
                            label, distance = self.face_mgr.recognize_face(face_encoding)
                            if label is not None:
                                name = customers_cache.get(str(label), "")
                                accuracy = max(0, min(100, (1 - distance) * 100))
                                label_text = f"{name} ({label}) - {accuracy:.0f}%"
                                color_rgb = (0, 255, 0)
                                self.out_queue.put(("recognized", label_text))
                            else:
                                label_text = f"Người Lạ - {(1-distance)*100:.0f}%"
                        except Exception as e:
                            print(f"Recognition error: {e}")
                    # Vẽ label bằng PIL để đẹp hơn
                    try:
                        rgb_display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb_display)
                        draw = ImageDraw.Draw(pil_img)
                        text_pos = (dx1, max(0, dy1 - 25))
                        bbox = draw.textbbox(text_pos, label_text, font=self.font_cache)
                        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=(0, 0, 0))
                        draw.text(text_pos, label_text, font=self.font_cache, fill=color_rgb)
                        display = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except Exception:
                        cv2.putText(display, label_text, (dx1, max(0, dy1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0])), 2)
                try:
                    rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    rgb = np.asarray(rgb, dtype=np.uint8)
                    self.out_queue.put(("frame", rgb))
                except Exception as e:
                    print(f"Queue put error: {e}")
                time.sleep(0.03)
        except Exception as e:
            self.out_queue.put(("error", str(e)))
            print("CameraThread error:", traceback.format_exc())
        finally:
            cap.release()
            self.out_queue.put(("stopped", None))

# ============ MAIN APP ============
class HostelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Quản Lý Nhà Trọ")
        self.root.geometry("1400x800")
        self.root.configure(bg="#f0f0f0")
        self.customer_db = ExcelDB(CUSTOMER_FILE)
        self.room_db = ExcelDB(ROOM_FILE)

        try:
            self.face_mgr = FaceManager()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Face recognition error: {e}")
            self.face_mgr = None

        self.cam_queue = queue.Queue()
        self.cam_stop_event = threading.Event()
        self.cam_thread = None
        self.capture_win = None
        self.capture_stop_event = None
        self.capture_preview_label = None
        self.capture_progress_label = None
        self.capture_thread = None
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        main_container = Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill=BOTH, expand=True)
        sidebar = Frame(main_container, width=200, bg="#2C3E50")
        sidebar.pack(side=LEFT, fill=Y)
        sidebar.pack_propagate(False)
        title_frame = Frame(sidebar, bg="#2C3E50", height=80)
        title_frame.pack(fill=X)
        title_frame.pack_propagate(False)
        Label(title_frame, text="QUẢN LÝ\nNHÀ TRỌ", bg="#2C3E50", fg="white",
              font=("Arial", 16, "bold"), justify=CENTER).pack(expand=True)
        menus = [
            ("📊 Thống Kê", self.show_stats),
            ("👥 Khách Thuê", self.show_customers),
            ("🏠 Phòng Trọ", self.show_rooms),
            ("📷 Camera", self.show_camera) ]

        for text, cmd in menus:
            btn = Button(sidebar, text=text, command=cmd, bg="#34495E", fg="white",
                         font=("Arial", 11), bd=0, padx=20, pady=15,
                         activebackground="#1ABC9C", activeforeground="white",
                         cursor="hand2", anchor=W)
            btn.pack(fill=X, pady=2)
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg="#1ABC9C"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg="#34495E"))

        self.content_area = Frame(main_container, bg="white")
        self.content_area.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)
        self.show_stats()

    def clear_content(self):
        for widget in self.content_area.winfo_children():
            widget.destroy()

    def show_stats(self):
        self.clear_content()
        Label(self.content_area, text="THÔNG TIN NHÀ TRỌ",
              font=("Arial", 20, "bold"), bg="white", fg="#2C3E50").pack(pady=10)
        info_frame = Frame(self.content_area, bg="white")
        info_frame.pack(fill=X, padx=20, pady=10)
        info_data = {
            "Tên trọ": "Nhà trọ hiện đại",
            "Chủ trọ": "Nhóm 15",
            "Địa chỉ": "Số 1 Võ Văn Ngân, Thủ Dức, TP.HCM",
            "Số điện thoại": "0987654321",
            "Ngày thành lập": "31/12/2025"  }
        for key, val in info_data.items():
            row = Frame(info_frame, bg="white")
            row.pack(fill=X, pady=5)
            Label(row, text=f"{key}:", font=("Arial", 11, "bold"),
                  bg="white", width=15, anchor=W).pack(side=LEFT, padx=5)
            Label(row, text=val, font=("Arial", 11),
                  bg="white", anchor=W).pack(side=LEFT, padx=5)

        stats_frame = Frame(self.content_area, bg="white")
        stats_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)
        df_cust = self.customer_db.read()
        df_room = self.room_db.read()
        total_cust = len(df_cust) if not df_cust.empty else 0
        total_rooms = len(df_room) if not df_room.empty else 0
        occupied = 0
        if not df_room.empty and 'room_id' in df_room.columns:
            for _, r in df_room.iterrows():
                rid = r.get('room_id', '')
                if self.customer_db.count_room_members(rid) > 0:
                    occupied += 1       
        stats_data = [
            ("Tổng khách", total_cust, "#3498DB"),
            ("Tổng phòng", total_rooms, "#E74C3C"),
            ("Phòng thuê", occupied, "#2ECC71"),
            ("Phòng trống", total_rooms - occupied, "#F39C12")   ]
        for text, value, color in stats_data:
            card = Frame(stats_frame, bg=color, relief=RAISED, bd=2)
            card.pack(side=LEFT, fill=BOTH, expand=True, padx=10)
            Label(card, text=str(value), font=("Arial", 36, "bold"),
                  bg=color, fg="white").pack(pady=(30, 5))
            Label(card, text=text, font=("Arial", 12),
                  bg=color, fg="white").pack(pady=(0, 30))

    # ============ CUSTOMERS ============
    def show_customers(self):
        self.clear_content()
        Label(self.content_area, text="QUẢN LÝ KHÁCH THUÊ",
              font=("Arial", 18, "bold"), bg="white", fg="#2C3E50").pack(pady=10)
        main_frame = Frame(self.content_area, bg="white")
        main_frame.pack(fill=BOTH, expand=True)
        form_frame = Frame(main_frame, bg="white", width=350)
        form_frame.pack(side=LEFT, fill=Y, padx=(10, 5))
        form_frame.pack_propagate(False)
        Label(form_frame, text="Thông tin khách", font=("Arial", 14, "bold"),
              bg="white").pack(pady=10)
        self.cust_entries = {}
        fields = [
            ("Mã khách:", "cust_id"),
            ("Họ tên:", "cust_name"),
            ("Số điện thoại:", "cust_phone"),
            ("Mã phòng:", "cust_room")]
        
        for label, key in fields:
            row = Frame(form_frame, bg="white")
            row.pack(fill=X, padx=10, pady=5)
            Label(row, text=label, bg="white", width=12, anchor=W).pack(side=LEFT)
            entry = Entry(row)
            entry.pack(side=LEFT, fill=X, expand=True)
            self.cust_entries[key] = entry

        for label, key in [("Ngày sinh:", "cust_dob"), ("Ngày đến:", "cust_start")]:
            row = Frame(form_frame, bg="white")
            row.pack(fill=X, padx=10, pady=5)
            Label(row, text=label, bg="white", width=12, anchor=W).pack(side=LEFT)
            date_entry = DateEntry(row, date_pattern='yyyy-mm-dd')
            date_entry.pack(side=LEFT, fill=X, expand=True)
            self.cust_entries[key] = date_entry

        btn_frame = Frame(form_frame, bg="white")
        btn_frame.pack(fill=X, padx=10, pady=20)
        btns = [
            ("➕ Thêm", self.add_customer, "#27AE60"),
            ("✏️ Sửa", self.edit_customer, "#3498DB"),
            ("🗑️ Xóa", self.delete_customer, "#E74C3C"),
            ("📷 Quét khuôn mặt", self.capture_face_with_preview, "#9B59B6") ]
        
        for text, cmd, color in btns:
            Button(btn_frame, text=text, command=cmd, bg=color, fg="white",
                   font=("Arial", 10, "bold"), cursor="hand2").pack(fill=X, pady=2)

        table_frame = Frame(main_frame, bg="white")
        table_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(5, 10))
        Label(table_frame, text="Danh sách khách", font=("Arial", 12, "bold"),
              bg="white").pack(pady=5)
        cols = ("Mã", "Tên", "Ngày sinh", "SĐT", "Ngày đến", "Phòng")
        self.cust_tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=25)
        for col in cols:
            self.cust_tree.heading(col, text=col)
            self.cust_tree.column(col, width=100, anchor=CENTER)
        scrollbar = Scrollbar(table_frame, orient=VERTICAL, command=self.cust_tree.yview)
        self.cust_tree.configure(yscrollcommand=scrollbar.set)
        self.cust_tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.cust_tree.bind("<<TreeviewSelect>>", self.on_customer_select)
        self.load_customers()

    def load_customers(self):
        if not hasattr(self, 'cust_tree'):
            return
        for item in self.cust_tree.get_children():
            self.cust_tree.delete(item)
        df = self.customer_db.read()
        if not df.empty:
            for _, row in df.iterrows():
                vals = (row.get('customer_id', ''), row.get('name', ''),
                        row.get('dob', ''), row.get('phone', ''),
                        row.get('start_date', ''), row.get('room_id', ''))
                self.cust_tree.insert('', 'end', values=vals)

    def add_customer(self):
        cid = self.cust_entries['cust_id'].get().strip()
        if not cid:
            messagebox.showerror("Lỗi", "Vui lòng nhập mã khách!")
            return
        df = self.customer_db.read()
        if not df.empty and cid in df['customer_id'].astype(str).values:
            messagebox.showerror("Lỗi", "Mã khách đã tồn tại!")
            return
        phone = self.cust_entries['cust_phone'].get().strip()
        if not phone.isdigit() or len(phone) != 10:
            messagebox.showerror("Lỗi", "Số điện thoại phải có 10 chữ số!")
            return
        try:
            dob = self.cust_entries['cust_dob'].get_date().strftime('%Y-%m-%d')
            start_date = self.cust_entries['cust_start'].get_date().strftime('%Y-%m-%d')
        except Exception:
            messagebox.showerror("Lỗi", "Ngày không hợp lệ!")
            return

        room_id = self.cust_entries['cust_room'].get().strip()
        room_df = self.room_db.read()

        if room_df.empty or room_id not in room_df['room_id'].astype(str).values:
            messagebox.showerror("Lỗi", f"Phòng {room_id} không tồn tại!")
            return
        try:
            max_p = int(room_df[room_df['room_id'].astype(str) == str(room_id)].iloc[0]['max_people'])
        except Exception:
            max_p = 0
        current = self.customer_db.count_room_members(room_id)
        if current >= max_p:
            return messagebox.showwarning("Phòng đầy", f"Phòng {room_id} đã đủ {max_p} người.")

        row = {
            'customer_id': cid,
            'name': self.cust_entries['cust_name'].get().strip(),
            'dob': dob,
            'phone': phone,
            'start_date': start_date,
            'room_id': room_id }
        
        if self.customer_db.append(row):
            self.update_room_status_after_change(room_id)
            messagebox.showinfo("Thành công", "Đã thêm khách!")
            self.load_customers()
            if hasattr(self, 'room_tree'):
                self.load_rooms()

    def edit_customer(self):
        cid = self.cust_entries['cust_id'].get().strip()
        if not cid:
            messagebox.showerror("Lỗi", "Vui lòng chọn khách!")
            return
        phone = self.cust_entries['cust_phone'].get().strip()
        if not phone.isdigit() or len(phone) != 10:
            messagebox.showerror("Lỗi", "Số điện thoại phải có 10 chữ số!")
            return
        try:
            dob = self.cust_entries['cust_dob'].get_date().strftime('%Y-%m-%d')
            start_date = self.cust_entries['cust_start'].get_date().strftime('%Y-%m-%d')
        except Exception:
            messagebox.showerror("Lỗi", "Ngày không hợp lệ!")
            return
        old_room = None
        df = self.customer_db.read()
        if not df.empty:
            res = df[df['customer_id'].astype(str) == str(cid)]
            if not res.empty:
                old_room = res.iloc[0].get('room_id', None)
        new_room = self.cust_entries['cust_room'].get().strip()
        if new_room != old_room:
            room_df = self.room_db.read()
            if room_df.empty or new_room not in room_df['room_id'].astype(str).values:
                return messagebox.showerror("Lỗi", f"Phòng {new_room} không tồn tại!")
            try:
                max_p = int(room_df[room_df['room_id'].astype(str) == str(new_room)].iloc[0]['max_people'])
            except Exception:
                max_p = 0
            current = self.customer_db.count_room_members(new_room)
            if current >= max_p:
                return messagebox.showwarning("Phòng đầy", f"Phòng {new_room} đã đủ {max_p} người.")
        updates = {
            'name': self.cust_entries['cust_name'].get().strip(),
            'dob': dob,
            'phone': phone,
            'start_date': start_date,
            'room_id': new_room }
        if self.customer_db.update('customer_id', cid, updates):
            if old_room:
                self.update_room_status_after_change(old_room)
            self.update_room_status_after_change(new_room)
            messagebox.showinfo("Thành công", "Đã cập nhật!")
            self.load_customers()
            if hasattr(self, 'room_tree'):
                self.load_rooms()

    def delete_customer(self):
        selected = self.cust_tree.selection()
        if not selected:
            messagebox.showerror("Lỗi", "Vui lòng chọn khách để xóa!")
            return
        vals = self.cust_tree.item(selected[0], "values")
        cid = vals[0]   # customer_id
        room_id = vals[5]  # cột thứ 6 là room_id trong Treeview
        # Xóa trong Excel khách hàng
        if self.customer_db.delete("customer_id", cid):
            # Xóa thư mục ảnh khuôn mặt
            user_folder = self.face_mgr._user_folder(cid)
            if user_folder.exists():
                import shutil
                shutil.rmtree(user_folder)
            # Cập nhật lại encodings
            if self.face_mgr.encodings_file.exists():
                try:
                    import pickle
                    with open(self.face_mgr.encodings_file, "rb") as f:
                        data = pickle.load(f)
                    encodings, ids = [], []
                    for enc, uid in zip(data["encodings"], data["ids"]):
                        if str(uid) != str(cid):
                            encodings.append(enc)
                            ids.append(uid)
                    with open(self.face_mgr.encodings_file, "wb") as f:
                        pickle.dump({"encodings": encodings, "ids": ids}, f)
                    self.face_mgr.known_encodings = encodings
                    self.face_mgr.known_ids = ids
                except Exception as e:
                    print("Error updating encodings:", e)
            # Cập nhật trạng thái phòng
            try:
                df_rooms = self.room_db.read()
                if not df_rooms.empty and room_id in df_rooms["room_id"].values:
                    df_rooms.loc[df_rooms["room_id"] == room_id, "status"] = "Trống"
                    self.room_db.write(df_rooms)
            except Exception as e:
                print("Error updating room status:", e)
            # Refresh lại giao diện
            self.load_customers()
            if hasattr(self, 'room_tree'):
                self.load_rooms()
            messagebox.showinfo("Thành công", f"Đã xóa khách {cid} và cập nhật phòng {room_id} thành Trống.")
        else:
            messagebox.showerror("Lỗi", "Không thể xóa khách này!")


    def on_customer_select(self, event):
        sel = self.cust_tree.selection()
        if not sel:
            return
        vals = self.cust_tree.item(sel[0], 'values')
        if not vals:
            return
        self.cust_entries['cust_id'].delete(0, END)
        self.cust_entries['cust_id'].insert(0, vals[0])
        self.cust_entries['cust_name'].delete(0, END)
        self.cust_entries['cust_name'].insert(0, vals[1])
        try:
            self.cust_entries['cust_dob'].set_date(vals[2])
        except Exception:
            pass
        try:
            self.cust_entries['cust_phone'].delete(0, END)
            self.cust_entries['cust_phone'].insert(0, vals[3])
        except Exception:
            pass
        try:
            self.cust_entries['cust_start'].set_date(vals[4])
        except Exception:
            pass
        try:
            self.cust_entries['cust_room'].delete(0, END)
            self.cust_entries['cust_room'].insert(0, vals[5])
        except Exception:
            pass
    # ============ CAPTURE ============
    def _update_capture_preview(self, frame_bgr):
        try:
            if self.capture_preview_label and self.capture_preview_label.winfo_exists():
                # Đảm bảo frame là numpy array hợp lệ
                if frame_bgr is None or frame_bgr.size == 0:
                    return
                # Resize frame
                preview_resized = cv2.resize(frame_bgr, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
                # Chuyển BGR sang RGB
                rgb = cv2.cvtColor(preview_resized, cv2.COLOR_BGR2RGB)
                # Đảm bảo là uint8
                rgb = np.asarray(rgb, dtype=np.uint8)
                # Tạo PIL Image
                img = Image.fromarray(rgb, mode='RGB')
                imgtk = ImageTk.PhotoImage(image=img)
                self.capture_preview_label.imgtk = imgtk
                self.capture_preview_label.config(image=imgtk)
        except Exception as e:
            print(f"Preview error: {e}")
            traceback.print_exc()
    
    def _update_capture_progress(self, current, total):
        try:
            if self.capture_progress_label and self.capture_progress_label.winfo_exists():
                progress_text = f"Đã chụp: {current}/{total} ảnh"
                self.capture_progress_label.config(text=progress_text)
        except Exception as e:
            print(f"Progress error: {e}")

    def capture_face_with_preview(self):
        cid = self.cust_entries.get('cust_id').get().strip()
        if not cid:
            messagebox.showerror("Lỗi", "Vui lòng nhập mã khách trước!")
            return
        if self.face_mgr is None:
            messagebox.showerror("Lỗi", "Face manager chưa khởi tạo!")
            return
        # Test camera trước khi mở cửa sổ
        test_cap = cv2.VideoCapture(0)
        if not test_cap.isOpened():
            messagebox.showerror("Lỗi Camera", "Không thể mở camera!\n\nKiểm tra:\n• Camera có được cắm?\n• Ứng dụng khác đang dùng camera?\n• Driver camera đã cài đặt?")
            return
        test_cap.release()
        time.sleep(0.5)
        # Tạo cửa sổ capture
        if self.capture_win and Toplevel.winfo_exists(self.capture_win):
            self.capture_win.lift()
            return
            
        win = Toplevel(self.root)
        win.title(f"Chụp ảnh cho khách {cid}")
        win.geometry(f"{PREVIEW_WIDTH+20}x{PREVIEW_HEIGHT+120}")
        win.resizable(False, False)       
        lbl = Label(win, bg="black", width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)
        lbl.pack(padx=10, pady=10)        
        progress_lbl = Label(win, text="Đang khởi động camera...", font=("Arial", 12, "bold"), fg="#2C3E50")
        progress_lbl.pack(pady=5)        
        btn_frame = Frame(win)
        btn_frame.pack(fill=X, padx=10, pady=5)        
        stop_ev = threading.Event()
        self.capture_stop_event = stop_ev
        self.capture_win = win
        self.capture_preview_label = lbl
        self.capture_progress_label = progress_lbl
        
        def close_capture():
            stop_ev.set()
            try:
                win.destroy()
            except:
                pass
            self.capture_win = None
            self.capture_preview_label = None
            self.capture_progress_label = None
        
        stop_btn = Button(btn_frame, text="Dừng", bg="#E74C3C", fg="white", font=("Arial", 10, "bold"), command=close_capture)
        stop_btn.pack(side=LEFT, padx=5)
        win.protocol("WM_DELETE_WINDOW", close_capture)

        def worker():
            try:
                saved = self.face_mgr.capture_images(
                    cid, 
                    num_samples=MAX_IMAGES_PER_USER,
                    preview_callback=self._update_capture_preview,
                    stop_event=stop_ev,
                    progress_callback=lambda c, t: self.root.after(0, lambda: self._update_capture_progress(c, t)))
                self.root.after(0, lambda: messagebox.showinfo("Hoàn tất", f"Đã lưu {saved} ảnh và tạo encodings cho khách {cid}.\nHệ thống sẵn sàng nhận diện!"))
                self.root.after(0, self.load_customers)
            except Exception as e:
                error_msg = f"Lỗi khi chụp ảnh:\n{str(e)}\n\nChi tiết:\n{traceback.format_exc()}"
                print(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Lỗi", error_msg))
            finally:
                self.root.after(0, close_capture)

        self.capture_thread = threading.Thread(target=worker, daemon=True)
        self.capture_thread.start()

    # ============ ROOMS ============
    def show_rooms(self):
        self.clear_content()
        Label(self.content_area, text="QUẢN LÝ PHÒNG TRỌ",
              font=("Arial", 18, "bold"), bg="white", fg="#2C3E50").pack(pady=10)
        main_frame = Frame(self.content_area, bg="white")
        main_frame.pack(fill=BOTH, expand=True)
        form_frame = Frame(main_frame, bg="white", width=360)
        form_frame.pack(side=LEFT, fill=Y, padx=(10,5))
        form_frame.pack_propagate(False)
        Label(form_frame, text="Thông tin phòng", font=("Arial", 14, "bold"), bg="white").pack(pady=8)
        self.room_entries = {}
        for label, key in [("Mã phòng:", "room_id"), ("Diện tích(m²):", "area"), ("Số người tối đa:", "max_people")]:
            row = Frame(form_frame, bg="white")
            row.pack(fill=X, padx=8, pady=6)
            Label(row, text=label, bg="white", width=14, anchor=W).pack(side=LEFT)
            ent = Entry(row)
            ent.pack(side=LEFT, fill=X, expand=True)
            self.room_entries[key] = ent
        row = Frame(form_frame, bg="white")
        row.pack(fill=X, padx=8, pady=6)
        Label(row, text="Trạng thái:", bg="white", width=14, anchor=W).pack(side=LEFT)
        status_var = StringVar(value="Trống")
        opt = OptionMenu(row, status_var, "Trống", "Đã thuê")
        opt.pack(side=LEFT, fill=X, expand=True)
        self.room_entries['status'] = status_var
        btn_frame = Frame(form_frame, bg="white")
        btn_frame.pack(fill=X, padx=8, pady=10)

        for text, cmd, color in [("➕ Thêm", self.add_room, "#27AE60"), 
                                  ("✏️ Sửa", self.edit_room, "#3498DB"), 
                                  ("🗑️Xóa", self.delete_room, "#E74C3C")]:
            
            Button(btn_frame, text=text, bg=color, fg="white", command=cmd, 
                   font=("Arial", 10, "bold")).pack(fill=X, pady=3)

        table_frame = Frame(main_frame, bg="white")
        table_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(5,10))
        Label(table_frame, text="Danh sách phòng", font=("Arial", 12, "bold"), bg="white").pack(pady=5)
        cols = ("Mã phòng", "Diện tích", "Số người tối đa", "Số người hiện tại", "Trạng thái")
        self.room_tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=20)
        for col in cols:
            self.room_tree.heading(col, text=col)
            self.room_tree.column(col, width=140, anchor=CENTER)
        self.room_tree.pack(fill=BOTH, expand=True)
        self.room_tree.bind("<<TreeviewSelect>>", self.on_room_select)
        self.load_rooms()

    def load_rooms(self):
        if not hasattr(self, 'room_tree'):
            return
        for item in self.room_tree.get_children():
            self.room_tree.delete(item)
        df = self.room_db.read()
        if not df.empty:
            for _, row in df.iterrows():
                rid = row.get('room_id', '')
                current = self.customer_db.count_room_members(rid)
                vals = (rid, row.get('area', ''), row.get('max_people', ''), current, row.get('status', ''))
                self.room_tree.insert('', 'end', values=vals)

    def add_room(self):
        rid = self.room_entries['room_id'].get().strip()
        if not rid:
            messagebox.showerror("Lỗi", "Nhập mã phòng!")
            return
        df = self.room_db.read()
        if not df.empty and rid in df['room_id'].astype(str).values:
            messagebox.showerror("Lỗi", "Mã phòng đã tồn tại!")
            return
        try:
            max_p = int(self.room_entries['max_people'].get().strip())
            if max_p < 0:
                raise ValueError()
        except Exception:
            messagebox.showerror("Lỗi", "Số người tối đa phải là số nguyên không âm!")
            return
        row = {
            'room_id': rid,
            'area': self.room_entries['area'].get().strip(),
            'max_people': max_p,
            'status': self.room_entries['status'].get().strip() or "Trống" }
        if self.room_db.append(row):
            messagebox.showinfo("Thành công", "Đã thêm phòng!")
            self.load_rooms()

    def edit_room(self):
        rid = self.room_entries['room_id'].get().strip()
        if not rid:
            messagebox.showerror("Lỗi", "Chọn phòng để sửa!")
            return
        df = self.room_db.read()
        if df.empty or rid not in df['room_id'].astype(str).values:
            messagebox.showerror("Lỗi", "Phòng không tồn tại!")
            return
        try:
            new_max = int(self.room_entries['max_people'].get().strip())
            if new_max < 0:
                raise ValueError()
        except Exception:
            messagebox.showerror("Lỗi", "Số người tối đa phải là số nguyên không âm!")
            return
        current = self.customer_db.count_room_members(rid)
        if new_max < current:
            return messagebox.showwarning("Không hợp lệ", f"Số người tối đa ({new_max}) < số người hiện tại ({current}).")
        updates = {
            'area': self.room_entries['area'].get().strip(),
            'max_people': new_max,
            'status': self.room_entries['status'].get().strip() }
        if self.room_db.update('room_id', rid, updates):
            messagebox.showinfo("Thành công", "Đã cập nhật phòng!")
            self.load_rooms()

    def delete_room(self):
        rid = self.room_entries['room_id'].get().strip()
        if not rid:
            messagebox.showerror("Lỗi", "Chọn phòng để xóa!")
            return
        if self.customer_db.count_room_members(rid) > 0:
            return messagebox.showwarning("Không thể xóa", "Phòng đang có khách.")
        if messagebox.askyesno("Xác nhận", f"Xóa phòng {rid}?"):
            if self.room_db.delete('room_id', rid):
                messagebox.showinfo("Thành công", "Đã xóa phòng!")
                self.load_rooms()

    def on_room_select(self, event):
        sel = self.room_tree.selection()
        if not sel:
            return
        vals = self.room_tree.item(sel[0], 'values')
        if not vals:
            return
        self.room_entries['room_id'].delete(0, END)
        self.room_entries['room_id'].insert(0, vals[0])
        self.room_entries['area'].delete(0, END)
        self.room_entries['area'].insert(0, vals[1])
        self.room_entries['max_people'].delete(0, END)
        self.room_entries['max_people'].insert(0, vals[2])
        try:
            self.room_entries['status'].set(vals[4])
        except Exception:
            pass

    def update_room_status_after_change(self, room_id):
        df = self.room_db.read()
        if df.empty:
            return
        mask = df['room_id'].astype(str) == str(room_id)
        if not mask.any():
            return
        current = self.customer_db.count_room_members(room_id)
        status = "Đã thuê" if current > 0 else "Trống"
        self.room_db.update('room_id', room_id, {'status': status})
        if hasattr(self, 'room_tree'):
            self.load_rooms()

    # ============ CAMERA ============
    def show_camera(self):
        self.clear_content()
        Label(self.content_area, text="CAMERA NHẬN DIỆN", 
            font=("Arial", 26, "bold"), bg="white", fg="#2C3E50").pack(pady=40)

        cam_frame = Frame(self.content_area, bg="white", width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)
        cam_frame.pack(pady=10, anchor=CENTER)
        cam_frame.pack_propagate(False)

        self.cam_label = Label(cam_frame, bg="black", width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)
        self.cam_label.pack()

        ctrl_frame = Frame(self.content_area, bg="white")
        ctrl_frame.pack(pady=10)

        btn_start = Button(ctrl_frame, text="BẬT CAM", command=self.start_camera, bg="#27AE60", fg="white", font=("Arial", 15, "bold"), width=12,height=1)
        btn_stop = Button(ctrl_frame, text="TẮT CAM", command=self.stop_camera, bg="#E74C3C", fg="white", font=("Arial", 15, "bold"), width=12,height=1)

        btn_start.grid(row=0, column=0, padx=20)
        btn_stop.grid(row=0, column=1, padx=20)

    def start_camera(self):
        if self.cam_thread and self.cam_thread.is_alive():
            messagebox.showinfo("Thông báo", "Camera đang chạy.")
            return
        # Test camera trước
        test_cap = cv2.VideoCapture(0)
        if not test_cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera!\nKiểm tra:\n- Camera có được cắm không?\n- Ứng dụng khác có đang dùng camera không?")
            return
        test_cap.release()
        time.sleep(0.5)       
        self.cam_stop_event.clear()
        self.cam_thread = CameraThread(self.cam_queue, self.cam_stop_event, self.face_mgr, self.customer_db)
        self.cam_thread.start()
        self.root.after(30, self._poll_camera_queue)

    def stop_camera(self):
        if self.cam_thread and self.cam_thread.is_alive():
            self.cam_stop_event.set()

    def _poll_camera_queue(self):
        try:
            while not self.cam_queue.empty():
                typ, data = self.cam_queue.get_nowait()
                if typ == "frame":
                    try:
                        # Đảm bảo data là RGB uint8 numpy array
                        if data is None or data.size == 0:
                            continue
                        data = np.asarray(data, dtype=np.uint8)
                        img = Image.fromarray(data, mode='RGB')
                        imgtk = ImageTk.PhotoImage(image=img)
                        self.cam_label.imgtk = imgtk
                        self.cam_label.config(image=imgtk)
                    except Exception as e:
                        print(f"Display frame error: {e}")
                        traceback.print_exc()
                elif typ == "recognized":
                    print("Recognized:", data)
                elif typ == "error":
                    messagebox.showerror("Camera error", data)
                elif typ == "stopped":
                    pass
        except Exception as e:
            print(f"Queue poll error: {e}")
        if self.cam_thread and self.cam_thread.is_alive():
            self.root.after(30, self._poll_camera_queue)
        else:
            try:
                self.cam_label.config(image='')
            except Exception:
                pass

    def on_closing(self):
        try:
            self.cam_stop_event.set()
        except Exception:
            pass
        try:
            if self.capture_stop_event:
                self.capture_stop_event.set()
        except Exception:
            pass
        self.root.destroy()
        
# ============ ENTRY POINT ============
if __name__ == "__main__":
    root = Tk()
    app = HostelApp(root)
    root.mainloop()