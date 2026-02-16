import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

dataset = "Hillvally"

# Load EXA_array (جمعیت نهایی)
EXA_array = np.load(f"1EXA_array{dataset}.npy", allow_pickle=True)

# Load x_train (دیتای ورودی)
x_train = np.load(f"D:/Drive_D/PhD/Term3/MetaHuri/seminar/NDEDA_CYB-main/split_73/train{dataset}.npy")

# ویژگی‌ها (X) و برچسب‌ها (y) را از x_train استخراج کن
X = x_train[:, 1:]  # همه ستون‌ها به جز ستون 0
y = x_train[:, 0]   # ستون 0 = labels

# محاسبه fitness برای هر فرد در جمعیت EXA_array
fitness_values = []

for individual in EXA_array:
    # انتخاب ویژگی‌های فرد (اگر 1 باشد، ویژگی انتخاب شده)
    selected_features = np.where(individual == 1)[0]

    # چاپ وضعیت انتخاب ویژگی‌ها
    print("Selected features for individual:", selected_features)

    if len(selected_features) == 0:
        # اگر هیچ ویژگی‌ای انتخاب نشده، حداقل یک ویژگی به‌طور تصادفی انتخاب کن
        individual[0] = 1  # حداقل یک ویژگی به‌طور تصادفی انتخاب شود
        selected_features = [0]  # برای جلوگیری از fitness = inf
        fitness_values.append(float('inf'))  # ذخیره‌سازی fitness
    else:
        # انتخاب ویژگی‌های انتخاب‌شده از X
        X_selected = X[:, selected_features]

        # استفاده از kNN برای ارزیابی عملکرد
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, X_selected, y, cv=5, scoring='accuracy')

        # fitness برابر با منفی دقیق‌ترین نتایج است (چون کمترین fitness مطلوب است)
        fitness_values.append(-np.mean(scores))  # نمره دقیق کثیرترین میانگین

# پیدا کردن بهترین فرد (کمترین fitness)
best_index = np.argmin(fitness_values)
best_solution = EXA_array[best_index]
best_fitness = fitness_values[best_index]

# اگر fitness برابر بی‌نهایت بود، تنظیم fitness پیش‌فرض
if best_fitness == float('inf'):
    best_fitness = "No valid solution found."

# نمایش بهترین فرد و تعداد ویژگی‌های انتخاب‌شده
print("Best Solution (0/1 Mask):", best_solution)
print("Best Fitness:", best_fitness)

# شماره ویژگی‌های انتخاب‌شده
selected_features = np.where(best_solution == 1)[0]
print("Selected Feature Indices:", selected_features)
print("Number of Selected Features:", len(selected_features))





# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score

# dataset = "Hillvally"

# # Load EXA_array (جمعیت نهایی)
# EXA_array = np.load(f"1EXA_array{dataset}.npy", allow_pickle=True)

# # Load x_train (دیتای ورودی)
# x_train = np.load(f"D:/Drive_D/PhD/Term3/MetaHuri/seminar/NDEDA_CYB-main/split_73/train{dataset}.npy")

# # ویژگی‌ها (X) و برچسب‌ها (y) را از x_train استخراج کن
# X = x_train[:, 1:]  # همه ستون‌ها به جز ستون 0
# y = x_train[:, 0]   # ستون 0 = labels

# # محاسبه fitness برای هر فرد در جمعیت EXA_array
# fitness_values = []

# for individual in EXA_array:
    # انتخاب ویژگی‌های فرد (اگر 1 باشد، ویژگی انتخاب شده)
#     selected_features = np.where(individual == 1)[0]

#     if len(selected_features) == 0:
#         fitness_values.append(float('inf'))  # اگر هیچ ویژگی‌ای انتخاب نشد، fitness برابر با بی‌نهایت می‌شود
#     else:
        # انتخاب ویژگی‌های انتخاب‌شده از X
#         X_selected = X[:, selected_features]

        # استفاده از kNN برای ارزیابی عملکرد
#         knn = KNeighborsClassifier(n_neighbors=5)
#         scores = cross_val_score(knn, X_selected, y, cv=5, scoring='accuracy')

#         # fitness برابر با منفی دقیق‌ترین نتایج است (چون کمترین fitness مطلوب است)
#         fitness_values.append(-np.mean(scores))  # نمره دقیق کثیرترین میانگین

# # پیدا کردن بهترین فرد (کمترین fitness)
# best_index = np.argmin(fitness_values)
# best_solution = EXA_array[best_index]
# best_fitness = fitness_values[best_index]

# # نمایش بهترین فرد و تعداد ویژگی‌های انتخاب‌شده
# print("Best Solution (0/1 Mask):", best_solution)
# print("Best Fitness:", best_fitness)

# # شماره ویژگی‌های انتخاب‌شده
# selected_features = np.where(best_solution == 1)[0]
# print("Selected Feature Indices:", selected_features)
# print("Number of Selected Features:", len(selected_features))
