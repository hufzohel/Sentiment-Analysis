import pandas as pd
import os
import urllib.request

def load_dataset(url, dest_folder="data", filename="dataset.csv"):
    """
    Tải dữ liệu từ URL nếu chưa có, sau đó trả về DataFrame.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        
    file_path = os.path.join(dest_folder, filename)
    
    if not os.path.exists(file_path):
        print(f"Đang tải dữ liệu từ {url}...")
        urllib.request.urlretrieve(url, file_path)
        print("Tải hoàn tất!")
    else:
        print("Dữ liệu đã tồn tại ở local, đang đọc file...")
        
    df = pd.read_csv(file_path)
    return df