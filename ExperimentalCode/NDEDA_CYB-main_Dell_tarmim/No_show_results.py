import numpy as np

dataset = "Hillvally"

min_fitness = np.load(f"1min_fitness{dataset}.npy")
unique_number = np.load(f"1unique_number{dataset}.npy")
running_time = np.load(f"1running_time{dataset}.npy")
EXA_array = np.load(f"1EXA_array{dataset}.npy", allow_pickle=True)

print("===== RESULTS =====")
print("Best Fitness:", min_fitness)
print("Unique Solutions:", unique_number)
print("Running Time (seconds):", running_time)
print("EXA_array shape:", EXA_array.shape)

# نمایش چند فرد اول برای نمونه
print("\nFirst 3 individuals of EXA_array:")
print(EXA_array[:3])

# شمارش ویژگی‌های انتخاب‌شده در بهترین فرد
pop_fit = np.load(f"1pop_fit{dataset}.npy", allow_pickle=True)
best_index = np.argmin(pop_fit)
best_solution = EXA_array[best_index]

print("\nBest solution (first 50 features):")
print(best_solution[:50])

print("\nNumber of selected features:", np.sum(best_solution))
