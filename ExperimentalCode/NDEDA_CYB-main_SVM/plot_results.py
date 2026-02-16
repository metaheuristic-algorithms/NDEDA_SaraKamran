import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# ---------- تنظیم نام دیتاست و مسیرها ----------
dataset = "zoo"
# dataset = "Hillvally"

base_dir = r"D:/Drive_D/PhD/Term3/MetaHuri/seminar/NDEDA_CYB-main"

path_exa = base_dir + f"/1EXA_array{dataset}.npy"
path_train = base_dir + f"/split_73/train{dataset}.npy"
path_minfit = base_dir + f"/1min_fitness{dataset}.npy"

print("Loading EXA_array from:", path_exa)
print("Loading train data from:", path_train)

EXA_array = np.load(path_exa, allow_pickle=True)
x_train = np.load(path_train)

X = x_train[:, 1:]
y = x_train[:, 0]

n_individuals, n_features = EXA_array.shape
print("EXA_array shape:", EXA_array.shape)
print("Train shape:", x_train.shape)

# ---------- محاسبه fitness، خطا و تعداد فیچرها برای هر فرد ----------

fitness_values = []
error_rates = []
num_features_list = []

print("\nEvaluating individuals... (این ممکنه چند دقیقه طول بکشه)")

for idx, individual in enumerate(EXA_array):
    # تبدیل به ماسک 0/1 با threshold مثل fit_train1
    mask = (individual >= 0.6).astype(int)
    selected_features = np.where(mask == 1)[0]

    if len(selected_features) == 0:
        f1 = 1.0         # خطا: بدترین حالت
        f2 = 0
        f = f1 + 1e-6 * f2
    else:
        X_selected = X[:, selected_features]
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, X_selected, y, cv=5, scoring="accuracy")
        f1 = 1 - np.mean(scores)       # error rate
        f2 = len(selected_features)    # تعداد فیچر
        f = f1 + 1e-6 * f2

    fitness_values.append(f)
    error_rates.append(f1)
    num_features_list.append(len(selected_features))

    if (idx + 1) % 10 == 0 or idx == n_individuals - 1:
        print(f"  evaluated {idx+1}/{n_individuals} individuals")

fitness_values = np.array(fitness_values)
error_rates = np.array(error_rates)
num_features_list = np.array(num_features_list)

# ---------- ۱) هیستوگرام خطای طبقه‌بندی ----------

plt.figure()
plt.hist(error_rates, bins=15)
plt.xlabel("Error rate (5-fold kNN)")
plt.ylabel("Count")
plt.title("Distribution of Error Rates in Final Population")
plt.tight_layout()

# ---------- ۲) فراوانی انتخاب فیچرها ----------

# ماسک باینری کل جمعیت
binary_mask = (EXA_array >= 0.6).astype(int)
feature_freq = binary_mask.sum(axis=0)  # چند بار هر فیچر انتخاب شده

plt.figure()
plt.bar(np.arange(n_features), feature_freq)
plt.xlabel("Feature index")
plt.ylabel("Selection count")
plt.title("Feature Selection Frequency in Final Population")
plt.tight_layout()

# ---------- ۳) رابطه تعداد فیچر و خطا (scatter) ----------

plt.figure()
plt.scatter(num_features_list, error_rates)
plt.xlabel("Number of selected features")
plt.ylabel("Error rate (5-fold kNN)")
plt.title("Error vs. Number of Selected Features")
plt.tight_layout()

# ---------- ۴) روند مینیمم fitness در طول نسل‌ها ----------

try:
    min_fitness_array = np.load(path_minfit, allow_pickle=True)
    # ممکنه یک لیست از مینیمم‌ها در هر نسل باشد
    min_fitness_array = np.array(min_fitness_array, dtype=float)

    plt.figure()
    plt.plot(min_fitness_array)
    plt.xlabel("Generation")
    plt.ylabel("Minimum fitness")
    plt.title("Evolution of Minimum Fitness Over Generations")
    plt.tight_layout()
except FileNotFoundError:
    print("\n[Warning] فایل 1min_fitnessHillvally.npy پیدا نشد، نمودار روند مین‌فیتنس رسم نشد.")

# ---------- نمایش همه نمودارها ----------
plt.show()

# ---------- گزارش خلاصه روی کنسول ----------

best_idx = int(np.argmin(fitness_values))
print("\n===== SUMMARY =====")
print("Best individual index     :", best_idx)
print("Best fitness (f)          :", fitness_values[best_idx])
print("Best error rate (f1)      :", error_rates[best_idx])
print("Best accuracy             :", 1 - error_rates[best_idx])
print("Num features (best)       :", num_features_list[best_idx])

best_mask = (EXA_array[best_idx] >= 0.6).astype(int)
print("Selected feature indices  :", np.where(best_mask == 1)[0])
