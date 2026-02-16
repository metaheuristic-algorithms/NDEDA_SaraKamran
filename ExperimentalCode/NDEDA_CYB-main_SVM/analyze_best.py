import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# dataset = "Hillvally"
# dataset = "zoo"
dataset = "WDBC"
# dataset = "ionosphere"

# 1) بارگذاری جمعیت نهایی
EXA_array = np.load(f"1EXA_array{dataset}.npy", allow_pickle=True)

# 2) بارگذاری داده‌ی آموزشی
# x_train = np.load(r"D:/Drive_D/PhD/Term3/MetaHuri/seminar/NDEDA_CYB-main/split_73/trainionosphere.npy")
# x_train = np.load(r"D:/Drive_D/PhD/Term3/MetaHuri/seminar/NDEDA_CYB-main/split_73/trainzoo.npy")
x_train = np.load(r"D:/Drive_D/PhD/Term3/MetaHuri/seminar/NDEDA_CYB-main/split_73/trainWDBC.npy")
# x_train = np.load(r"D:/Drive_D/PhD/Term3/MetaHuri/seminar/NDEDA_CYB-main/split_73/trainHillvally.npy")

X = x_train[:, 1:]  # فیچرها
y = x_train[:, 0]   # لیبل‌ها

fitness_values = []
error_rates = []
num_features_list = []

for individual in EXA_array:
    # 🔹 مثل خود الگوریتم: threshold روی 0.6
    mask = (individual >= 0.6).astype(int)
    selected_features = np.where(mask == 1)[0]

    # اگر هیچ فیچری انتخاب نشد → بدترین حالت
    if len(selected_features) == 0:
        f1 = 1.0          # error rate = 1
        f2 = 0            # تعداد فیچر
        f = f1 + 1e-6*f2  # مثل fit_train1
    else:
        X_selected = X[:, selected_features]
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, X_selected, y, cv=5, scoring="accuracy")

        f1 = 1 - np.mean(scores)     # error rate
        f2 = len(selected_features)  # تعداد فیچر
        f = f1 + 1e-6 * f2           # همان فرم مقاله/کد

    fitness_values.append(f)
    error_rates.append(f1)
    num_features_list.append(len(selected_features))

for i, ind in enumerate(EXA_array[:10]):
    mask = (ind >= 0.6).astype(int)
    selected = np.where(mask == 1)[0]
    print(f"Solution {i}: #features={len(selected)}, features={selected}")
    
# 3) پیدا کردن بهترین فرد
best_index = int(np.argmin(fitness_values))
best_solution_cont = EXA_array[best_index]
best_mask = (best_solution_cont >= 0.6).astype(int)
best_selected_features = np.where(best_mask == 1)[0]

best_fitness = fitness_values[best_index]
best_error = error_rates[best_index]
best_num_features = num_features_list[best_index]
best_accuracy = 1 - best_error

print("===== BEST INDIVIDUAL =====")
print("=======Dataset:", dataset)
print("Index in population      :", best_index)
print("Best fitness (f)         :", best_fitness)
print("Error rate (f1)          :", best_error)
print("Accuracy                 :", best_accuracy)
print("Number of selected feats :", best_num_features)
print("Selected feature indices :", best_selected_features)

print("\nRaw continuous vector (first 20 dims):")
print(best_solution_cont[:20])
