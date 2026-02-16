
import os
import numpy as np
import pandas as pd

# 1) مسیر فایل Training بدون نویز از UCI
# raw_path = r"D:\Drive_D\PhD\Term3\MetaHuri\seminar\NDEDA_CYB-main\data\Hill_Valley_without_noise_Training.data"
raw_path = r"D:\\Drive_D\PhD\Term3\MetaHuri\seminar\NDEDA_CYB-main\data\hill_valley\Hill_Valley_without_noise_Training.data"
# raw_path = r"D:\\Drive_D\PhD\Term3\MetaHuri\seminar\NDEDA_CYB-main\data\WBCD\wdbc.data"
# raw_path = r"D:\\Drive_D\PhD\Term3\MetaHuri\seminar\NDEDA_CYB-main\data\ionosphere\ionosphere.data"

print("Reading raw file from:", raw_path)

# ⚠️ این بار header=0 می‌گذاریم تا سطر اول (X1,...,class) هدر باشد، نه دیتا
# اغلب این فایل‌ها با کاما جدا شده‌اند، پس sep="," هم می‌گذاریم
df = pd.read_csv(raw_path, header=0, sep=",")  # اگر خطا داد، sep="," را بردار

print("Raw DataFrame shape:", df.shape)
print("Columns:", df.columns.tolist()[:5], "...")

# 2) آخرین ستون = label (class)، بقیه = feature ها
X = df.iloc[:, :-1].values   # همه ستون‌ها به جز آخری
y = df.iloc[:, -1].values    # فقط ستون آخر

print("X shape:", X.shape)
print("y shape:", y.shape)

# 3) تبدیل به فرم مورد نیاز NDEDA:
# ستون 0 = label, ستون‌های 1.. = features
y = y.reshape(-1, 1)
data_combined = np.hstack([y, X])   # شکل: (n_samples, 101)

print("Combined shape (label + features):", data_combined.shape)

# 4) مسیر ذخیره npy
save_dir = r"D:\Drive_D\PhD\Term3\MetaHuri\seminar\NDEDA_CYB-main\split_73"
os.makedirs(save_dir, exist_ok=True)

dataset_name = "Hillvally"  # همون اسمی که بعداً به کد می‌دیم
save_path = os.path.join(save_dir, f"train{dataset_name}.npy")

print("Saving npy to:", save_path)
np.save(save_path, data_combined.astype(np.float32))

print("✅ Done.")






















