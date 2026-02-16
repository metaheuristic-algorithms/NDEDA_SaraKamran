import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import os

# تنظیمات اولیه
dataset = "WDBC"
strategies = ['DE_rand_1', 'DE_best_1', 'DE_current-to-best_1']

# ۲) بارگذاری داده‌ی آموزشی (مطمئن شوید مسیر درست است)
data_path = r"D:/Drive_D/PhD/Term3/MetaHuri/seminar/behbood/NDEDA_CYB-main_version/split_73/trainWDBC.npy"

if not os.path.exists(data_path):
    print(f"Error: Data file not found at {data_path}")
else:
    x_train = np.load(data_path)
    X = x_train[:, 1:]  # ویژگی‌ها
    y = x_train[:, 0]   # لیبل‌ها

    print(f"{'Strategy':<20} | {'Best Fitness':<12} | {'Accuracy':<10} | {'Features':<8}")
    print("-" * 65)

    for strat in strategies:
        file_name = f"{strat}_EXA_array_{dataset}.npy"
        
        if not os.path.exists(file_name):
            print(f"File {file_name:<20} | Not Found")
            continue

        EXA_array = np.load(file_name, allow_pickle=True)
        fitness_values = []
        
        for individual in EXA_array:
            mask = (individual >= 0.6).astype(int)
            selected_features = np.where(mask == 1)[0]

            if len(selected_features) == 0:
                f = 1.0 # بدترین حالت اگر هیچ ویژگی انتخاب نشود
            else:
                X_selected = X[:, selected_features]
                # استفاده از SVM هماهنگ با بخش Training
                clf = SVC(kernel='rbf', random_state=42) 
                scores = cross_val_score(clf, X_selected, y, cv=5)
                
                error_rate = 1 - np.mean(scores)
                num_feats = len(selected_features)
                # فرمول فیتنس: خطا + جریمه بسیار کوچک برای تعداد ویژگی
                f = error_rate + 0.000001 * num_feats

            fitness_values.append(f)

        # پیدا کردن بهترین فرد در این استراتژی خاص
        best_idx = np.argmin(fitness_values)
        best_fitness = fitness_values[best_idx]
        
        # استخراج ویژگی‌های بهترین فرد جهت محاسبه دقت خالص
        best_ind = EXA_array[best_idx]
        best_mask = (best_ind >= 0.6).astype(int)
        best_feats = np.where(best_mask == 1)[0]
        num_best_feats = len(best_feats)

        # محاسبه دقت واقعی (حذف اثر جریمه Penalty)
        # Fitness = (1 - Accuracy) + (0.000001 * n)
        # Accuracy = 1 - (Fitness - (0.000001 * n))
        actual_accuracy = 1 - (best_fitness - (0.000001 * num_best_feats))

        # چاپ ردیف جدول برای مقایسه
        print(f"{strat:<20} | {best_fitness:<12.6f} | {actual_accuracy:<10.4f} | {num_best_feats:<8}")
        
        


# این روش برام تعداد ویژگی متفاوت میده ولی با دقت 1
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score
# import numpy as np
# from sklearn.svm import SVC # تغییر از KNN به SVM
# from sklearn.model_selection import cross_val_score

# # dataset = "Hillvally"
# # dataset = "zoo"
# dataset = "WDBC"
# # dataset = "ionosphere"

# # 1) بارگذاری جمعیت نهایی
# # EXA_array = np.load(f"1EXA_array{dataset}.npy", allow_pickle=True)
# strategies = ['DE_rand_1', 'DE_best_1', 'DE_current-to-best_1']

# # 2) بارگذاری داده‌ی آموزشی
# # x_train = np.load(r"D:/Drive_D/PhD/Term3/MetaHuri/seminar/NDEDA_CYB-main/split_73/trainionosphere.npy")
# # x_train = np.load(r"D:/Drive_D/PhD/Term3/MetaHuri/seminar/NDEDA_CYB-main/split_73/trainzoo.npy")
# # x_train = np.load(r"D:/Drive_D/PhD/Term3/MetaHuri/seminar/NDEDA_CYB-main/split_73/trainWDBC.npy")
# x_train = np.load(r"D:/Drive_D/PhD/Term3/MetaHuri/seminar/behbood/NDEDA_CYB-main_version/split_73/trainWDBC.npy")
# # x_train = np.load(r"D:/Drive_D/PhD/Term3/MetaHuri/seminar/NDEDA_CYB-main/split_73/trainHillvally.npy")



# X = x_train[:, 1:]  # فیچرها
# y = x_train[:, 0]   # لیبل‌ها

# fitness_values = []
# error_rates = []
# num_features_list = []

# for strat in strategies:
#     file_name = f"{strat}_EXA_array_{dataset}.npy"
#     try:
#         EXA_array = np.load(file_name, allow_pickle=True)
#         print(f"\n--- Results for Strategy: {strat} ---")
#     except:
#         print(f"File {file_name} not found.")
#         continue

#     fitness_values = []
#     
#     for individual in EXA_array:
#         mask = (individual >= 0.6).astype(int)
#         selected_features = np.where(mask == 1)[0]

#         if len(selected_features) == 0:
#             f = 1.0
#         else:
#             X_selected = X[:, selected_features]
#             # هماهنگ با تابع fit_train1 در کد اصلی
#             clf = SVC(kernel='rbf') 
#             scores = cross_val_score(clf, X_selected, y, cv=5)
#             
#             f1 = 1 - np.mean(scores)
#             f2 = len(selected_features)
#             f = f1 + 0.000001 * f2

#         fitness_values.append(f)
#         
#         

#     # پیدا کردن بهترین فرد در این استراتژی
#     best_idx = np.argmin(fitness_values)
#     best_ind = EXA_array[best_idx]
#     best_mask = (best_ind >= 0.6).astype(int)
#     best_feats = np.where(best_mask == 1)[0]
#     
#     print(f"Best Fitness: {fitness_values[best_idx]:.6f}")
#     print(f"Accuracy: {1 - (fitness_values[best_idx] // 1):.4f}") # تقریب دقت
#     print(f"Num Features: {len(best_feats)}")
#     print(f"Features: {best_feats}")
#     
#     
# این کد را چطور درست کنم    
        
    