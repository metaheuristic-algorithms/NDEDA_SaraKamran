import os
import sys
# os.chdir('D:/Drive_D/PhD/Term3/MetaHuri/seminar/NDEDA_CYB-main')



from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
from itertools import chain
import array
import random
import json
import numpy as np
from math import sqrt
from deap import algorithms
from deap import base
import math,time
from deap import benchmarks
from deap import creator
from deap import tools
from operator import mul
from functools import reduce
import alg_single_LBPADE
import numpy.matlib
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
from collections import Counter
import geatpy as ea
from single_diverse2 import produce_unique_individuals#,pre_selection
import sys,saveFile


# create random vector in [low, up]
def uniform(low, up, size=None):####generate a matrix of the range of variables
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

# Finding all indices of a value in a list
# Find out which people have the same binary pattern.
def findindex(org, x):
    result = []
    for k,v in enumerate(org): 
        if v == x:
            result.append(k)
    return result

# find all substrings from a bigger string 
# Finding the position of '1's in a binary string
def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)

# Fitnesss=Error Rate+ landa*|feature size|
# find Fitness Value for X_i   selecting features with Error rate and #features selected
def fit_train1(x1, train_data): #X1=individual with continuous vector
    x = np.zeros((1,len(x1)))
    for ii in range(len(x1)):
        x[0,ii] = x1[ii]
    x = random.choice(x)
    #tau = 0.5
    x = 1 * (x >= 0.6) #if the value of a dimention of vector is > 0.6 that feature will selected (convert binary values)
    if np.count_nonzero(x) == 0:
        f1 = 1###error_rate
        f2 = 1
    else:
    #prepare for training and also use label in dataset
     x = "".join(map(str, x))  # transfer the array form to string in order to find the position of 1
     value_position = np.array(list(find_all(x, '1'))) + 1  # cause the label in the first column in training data
     value_position = np.insert(value_position, 0, 0)  # insert the column of label
     tr = train_data[:, value_position]
     # Error rate 
     clf = KNeighborsClassifier(n_neighbors = 5)
     scores = cross_val_score(clf, tr[:,1:],tr[:,0], cv = 5)
     f1 = np.mean(1 - scores) # error_rate
     #subset feature size
     f2 = len(value_position)-1 #number of feature selected
    # f1+ landa*|feature size|
    f = f1 + 0.000001 * f2
    return f


# Implementation of the k-nearest neighbor (kNN) classifier.
def kNNClassify(newInput, dataSet, labels, k):
    """Calculates the distance (here Euclidean distance) of the new sample (newInput) to all training samples (dataSet).
       Finds the $k$ nearest neighbor and determines the label of the new sample 
       based on the majority vote of the neighbor labels.
       """
    numSamples = dataSet.shape[0]   
    diff = np.tile(newInput, (numSamples, 1)) - dataSet 
    squaredDiff = diff ** 2 
    squaredDist = squaredDiff.sum(axis = 1)   
    distance = squaredDist ** 0.5  
    sortedDistIndices = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sorted_ClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_ClassCount[0][0]


def evaluate_test_data(x, train_data, test_data):
    """
    Objective: Evaluate the final performance of a solution (individual x) on test data (test_data).
    Output: Error rate (f_1) on the test set and feature size ratio (f_2) (number of selected features divided by total features).
    """
    x = 1 * (x >= 0.6)
    x = "".join(map(str, x))  # transfer the array form to string in order to find the position of 1
    value_position = np.array(list(find_all(x, '1'))) + 1  # cause the label in the first column in training data
    value_position = np.insert(value_position, 0, 0)  # insert the column of label
    te = test_data[:, value_position]#####testing data including label in the first colume
    tr = train_data[:, value_position]#####training data including label in the first colume too
    wrong = 0
    for i12 in range(len(te)):
        testX = te[i12,1:]
        dataSet = tr[:,1:]
        labels = tr[:,0]
        outputLabel = kNNClassify(testX, dataSet, labels, 5)
        # print(outputLabel,te[i12,0])
        if outputLabel != te[i12,0]:
            wrong = wrong + 1
    f1 = wrong/len(te)
    f2 = (len(value_position) - 1) / (test_data.shape[1] - 1)
    return f1, f2


# Removing duplicate individuals based on binary representation (diversity)
def more_confidence(EXA, index_of_objectives):
    """
    When several individuals have exactly the same subset of features after 
    thresholding, this function tells which individual chose that subset “more confidently”:
    That is, the distance between their values ​​is greater than 0.6 (sum of confidence).
    -----------
    Confidence: For a floating point value $b$ and threshold $a=0.6$:
    If $b >= 0.6$ (selected), the confidence is ${b - a}/{1 - a}$. (The closer to 1, the better)
    If $b < 0.6$ (not selected), the confidence is ${a - b}/{a}. (The closer to 0, the better)
    The solution with the highest confidence sum is chosen.
    """
    a = 0.6
    cr = np.zeros((len(index_of_objectives),1))
    for i in range(len(index_of_objectives)):###the number of indexes
        temp = 0
        object = EXA[index_of_objectives[i]]
        for ii in range(len(object)):###the number of features
           b = object[ii]
           if b > a:  con = (b - a) / (1 - a)
           else:      con = (a - b) / (a)
           temp = con + temp
        cr[i,0] = temp
    sorting = np.argsort(-cr[:,0])####sorting from maximum to minimum
    index_one = index_of_objectives[sorting[0]]
    return index_one


def delete_duplicate(EXA):####list
    """
    Goal: Eliminate duplicate solutions in terms of binary feature subsets.
    Function: Converts floating-point solutions to equivalent "binary" feature subsets, 
    then keeps only one "unique representative" for each feature subset. If there are "multiple solutions" 
    for a feature subset, keeps the solution with the highest confidence using "more_confidence".
    """
    EXA1 = []
    EXA_array = np.array(EXA)
    all_index = []
    for i0 in range(EXA_array.shape[0]):
       x = 1 * (EXA_array[i0,:] >= 0.6)
       x = "".join(map(str, x))  # transfer the array form to string in order to find the position of 1
       all_index.append(x)##store all individuals who have changed to 0 or 1
    single_index = set(all_index)####find the unique combination
    single_index = list(single_index)####translate it's form in order to following operating.
    for i1 in range(len(single_index)):
       index_of_objectives = findindex(all_index, single_index[i1])##find the index of each unique combination
       if len(index_of_objectives) == 1:
          for i2 in range(len(index_of_objectives)):
             EXA1.append(EXA[index_of_objectives[i2]])
       else:####some combination have more than one solutions.here may have duplicated solutions
           index_one = more_confidence(EXA, index_of_objectives)
           EXA1.append(EXA[index_one])
    # print(EXA1)
    return EXA1


# def generate(size, pmin, pmax, smin, smax):
#     part = creator.Particle(numpy.random.uniform(pmin, pmax, size))
#     part.speed = numpy.random.uniform(smin, smax, size)
#     part.smin = smin
#     part.smax = smax
#     return part


# not used
# def mutDE(y, a, b, c, f):###mutation:DE/rand/1; if a is the best one, it will change as DE/best/1
#     for i in range(len(y)):
#         y[i] = a[i] + f*(b[i]-c[i])
#     return y


#nei= number of nearest knighbers for selecting best person ii= the index of person which will mutation offspring= current generation
#dis= distance between persons in population
def proposed_mutation2(dis,offspring,ii,nei,gen,fit_num,Max_FES):#########if best solution is larger than a half, select solutions from current niche
    ss = np.argsort(dis) #مرتب سازی بر اساس فاصله افراد
    niche_offspring = [offspring[t] for t in ss[1:nei]]###use to compare the fitness
    member = offspring[ii]   # فردی که جهش روی آن اعمال می‌شود
    s = [(member.fitness.value>=t.fitness.value)*1 + (member.fitness.value<t.fitness.value)*0 for t in niche_offspring] # یک لیست از مقادیر 0 یا 1 است که نشان می‌دهد آیا فیتنس فرد فعلی بزرگتر از فرد نیش است یا خیر
    sum_abs = sum(s) # تعداد افرادی است که فیتنس بالاتری نسبت به فرد فعلی دارند.
    #f_mu=(fit_num<=(0.8*Max_FES))*((sum_abs/(nei))*0.8+0.1)+(fit_num>(0.8*Max_FES))*(((sum_abs/(nei))* 0.8+0.1)*0.001)
    # f_mu= 0.5
    f_mu= 0.3 # # مقدار پیش‌فرض فاکتور مقیاس (برای جهش)
    pop_fit_whole = [ind.fitness.value for ind in offspring]
    niche_offspring1 = [offspring[t] for t in ss[:nei]] ## انتخاب n همسایه نزدیک
    pop_fit = [ind.fitness.value for ind in niche_offspring1]
    if sum_abs < 4:######from whole population choose nbest, while from niche randomly choose other two solutions کنام ضعیفه و باید از بهترین فرد کل جمعیت استفاده شود
        min_index = np.argwhere(pop_fit_whole == min(pop_fit_whole))
        min_one = random.choice(min_index)
        min_one = random.choice(min_one)
        nbest = offspring[min_one] # انتخاب بهترین فرد از جمعیت
        offspring1 = [niche_offspring1[t] for t in range(len(niche_offspring1))]  ###use to compare the fitness
    else:##from niche choose solution as nbest
        min_index = np.argwhere(pop_fit == min(pop_fit))
        min_one = random.choice(min_index)
        min_one = random.choice(min_one)
        nbest = niche_offspring1[min_one] # انتخاب بهترین فرد از نیش
        offspring1 = [offspring[t] for t in range(len(offspring))]  ###use to compare the fitness
    if member in offspring1: # حذف فردهای تکراری جهت جلوگیری از بازگشت به همان فردها و افزایش تنوع در جمعیت انجام می‌شود
            offspring1.remove(member) #جلوگیری از انتخاب دوباره خود member یا nbest.
    if nbest in offspring1:
            offspring1.remove(nbest)
    # تولید فرد جدید با استفاده از جهش    
    in1, in2 = random.sample(offspring1, 2)
    y_new = toolbox.clone(member)
    for i2 in range(len(y_new)):
               y_new[i2] = member[i2] + f_mu * (nbest[i2] - member[i2])+ f_mu  * (in1[i2] - in2[i2]) # = Eq2
    return y_new, nbest


##cxBinomial(offspring[ii],y_new,0.5)###crossover Binomial Crossover
def cxBinomial(x, y, cr):#####binary crossover  in main cr=0.7
    y_new = toolbox.clone(y)  # کپی کردن فرد دوم (والد دوم همان بردار جهش یافته) برای ایجاد فرد جدید
    size = len(x) # تعداد ویژگی‌ها (طول فرد)
    index = random.randrange(size) # انتخاب یک ویژگی به‌طور تصادفی برای شروع
    for i in range(size):
        #با احتمال CR، ویژگی از فرد دوم انتخاب می شود        
        # i==index Guaranteed to select at least one trait from the mutant individual
        if i == index or random.uniform(0, 1) <= cr: #   اگر به‌شکل تصادفی انتخاب شد یا به‌طور ویژه ویژگی شروع شد
            y_new[i] = y[i] # ویژگی از فرد دوم انتخاب می‌شود
            # y_new[i] = 0.8*(1- y[i])
        else:
            y_new[i] = x[i] # در غیر این صورت ویژگی از فرد اول (والد اول) انتخاب می‌شود
    return y_new



def continus2binary(x):
    for i in range(len(x)):
            # tau = 0.5
            if x[i] >= 0.6:
                x[i] = 1.0
            else:
                x[i] = 0.0
    return x


def hamming_distance(s, s0):
    """Return the Hamming distance between equal-length sequences"""
    s1 = toolbox.clone(s)
    s2 = toolbox.clone(s0)
    s3 = continus2binary(s1)
    s4 = continus2binary(s2)
    if len(s3) != len(s4):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s3, s4))

# not used
# def euclidean_distance(x1,x2):
#     s1 = toolbox.clone(x1)
#     s2 = toolbox.clone(x2)
#     s1 = np.array(s1)
#     s2 = np.array(s2)
#     temp = sum((s1-s2)**2)
#     temp1 = np.sqrt(temp)
#     return temp1



# def xor(a,b):
#     xor_value = (1-a)*b+ a*(1-b)
#     return xor_value


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))####minimise an objectives
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def main(seed,x_train):
    random.seed(seed)
    NDIM = x_train.shape[1] - 1
    ee = 1/x_train.shape[0] # ابعاد مسئله (تعداد ویژگی‌ها)
    BOUND_LOW, BOUND_UP = 0.0, 1.0 # محدودیت‌های متغیر
    NGEN = 100###the number of generation # تعداد نسل‌ها
    if NDIM < 300:
        MU = NDIM ####the number of particle #   # اندازه جمعیت   تعداد افراد برابر با تعداد ویژگی‌ها است
    else:
        MU = 300  #####bound to 300
    Max_FES = MU * 100 # حداکثر ارزیابی‌های مجاز
    nei = 9
    min_fitness = []
    unique_number = []
    # toolbox.register("attr_float", bytes, BOUND_LOW, BOUND_UP, NDIM)
    # تابع برای تولید افراد به صورت تصادفی    
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)  #####dertemine the way of randomly generation and gunrantuu the range
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)  ###fitness # ایجاد فرد
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ##particles # ایجاد جمعیت
    toolbox.register("evaluate", fit_train1, train_data= x_train) # ارزیابی fitness
    toolbox.register("select", alg_single_LBPADE.selNS)##alg4_NSGA2  # انتخاب از الگوریتم انتخاب NSGA2
    toolbox.register("select1", alg_single_LBPADE.selection_compared_with_nearest) # انتخاب با مقایسه نزدیک‌ترین همسایه‌ها
    # ... (مقداردهی اولیه جمعیت و ارزیابی)    
    offspring = toolbox.population(n=MU)
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)#####toolbox.evaluate = fit_train
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.value = fit
    pop_fit = [ind.fitness.value for ind in offspring] # دریافت fitness تمامی افراد
    min_fitness.append(min(pop_fit)) # ثبت بهترین fitness
    fit_num = len(offspring) # تعداد افراد در جمعیت
    #offspring = toolbox.select(offspring, len(offspring))
    pop_surrogate = delete_duplicate(offspring) # حذف افراد تکراری برای حفظ تنوع
    unique_number.append(len(pop_surrogate)) # تعداد افراد یکتا
    dis = np.zeros((MU,MU)) # فاصله بین افراد
 # محاسبه فاصله‌ها (فاصله همینگ)    # محاسبه فاصله‌ها بین تمامی افراد     
    for i in range(MU):
        for j in range(MU):
            dis[i,j] = hamming_distance(offspring[i],offspring[j])/NDIM
            #dis[i, j] = euclidean_distance(offspring[i], offspring[j]) / NDIM
    for gen in range(1, NGEN): # حلقه برای نسل‌های مختلف
        pop_new = toolbox.clone(offspring) # کپی جمعیت فعلی
        for ii in range(len(offspring)):#####upate the whole population  # بروزرسانی تمام افراد
            y_new,nbest= proposed_mutation2(dis[ii, :],offspring,ii,nei,gen,fit_num,Max_FES) # 1. جهش با رویکرد نیش‌گذاری
            for i_z in range(len(y_new)):
                    if y_new[i_z] > 1:
                        y_new[i_z] = nbest[i_z]
                    if y_new[i_z] < 0:
                        y_new[i_z] = nbest[i_z]
            ss = np.argsort(dis[ii, :])
            # pop_new[ii] = cxBinomial(offspring[ii],y_new,0.5)###crossover
            pop_new[ii] = cxBinomial(offspring[ii],y_new,0.7)###crossover  # 2. کراس‌اوور دوجمله‌ای
            del pop_new[ii].fitness.values###delete the fitness
        # pop_new = produce_diverse_individuals(pop_new, pop_non)
        ##################################################
        pop_unique = delete_duplicate(offspring) # حذف افراد تکراری برای حفظ تنوع
        print(f"Number of unique individuals in generation {gen}: {len(pop_unique)}") # چاپ تعداد افراد یکتا
        
        if NDIM <= 500:
           pop_new,pop_unique = produce_unique_individuals(pop_new,offspring,dis,pop_unique,nei) # تولید افراد یکتا
        pop_surrogate.extend(pop_unique) # اضافه کردن افراد یکتا به جمعیت
        pop_surrogate = delete_duplicate(pop_surrogate)  # حذف دوباره افراد تکراری
        unique_number.append(len(pop_surrogate)) # تعداد افراد یکتا را ذخیره می‌کنیم
        invalid_ind = [ind for ind in pop_new if not ind.fitness.valid]  # پیدا کردن افراد نامعتبر
        fitne = toolbox.map(toolbox.evaluate, invalid_ind) # ارزیابی fitness
        for ind, fit1 in zip(invalid_ind, fitne):
            ind.fitness.value = fit1 # اختصاص دادن fitness جدید به افراد
        # Select the next generation population
        # انتخاب جمعیت بعدی        
        fit_num = fit_num + len(offspring)
        pop_mi = pop_new + offspring # ترکیب جمعیت والدین و فرزندان
        pop1 = delete_duplicate(pop_mi)  # حذف افراد تکراری
        offspring = toolbox.select(pop1, MU,ee) # انتخاب N فرد با استفاده از استراتژی انتخاب
        #offspring = toolbox.select1(offspring, pop_new)
        pop_fit = [ind.fitness.value for ind in offspring]######selection from author # دریافت fitness تمام افراد
        min_fitness.append(min(pop_fit)) # ثبت بهترین fitness در هر نسل
        # offspring = toolbox.select(pop_new + offspring, MU)
        # محاسبه فاصله‌ها        
        for i in range(MU):
            for j in range(MU):
                dis[i, j] = hamming_distance(offspring[i], offspring[j]) / NDIM
                #dis[i, j] = euclidean_distance(offspring[i], offspring[j]) / NDIM
        ##########################################################new
        if fit_num > Max_FES:
            break
    return offspring,min_fitness,unique_number


if __name__ == "__main__":
    
    # sys.argv = ["main_NDEDA_public.py", "Hillvally", "1"]
    # sys.argv = ["main_NDEDA_public.py", "zoo", "1"]
    sys.argv = ["main_NDEDA_public.py", "WDBC", "1"]
    # sys.argv = ["main_NDEDA_public.py", "ionosphere", "1"]
    dataset_name = str(sys.argv[1])   # انتظار داریم Hillvally باشد
    seed = str(sys.argv[2])

    base_dir = r"D:\Drive_D\PhD\Term3\MetaHuri\seminar\NDEDA_CYB-main\split_73"
    folder1 = os.path.join(base_dir, f"train{dataset_name}.npy")
    print("Loading training data from:", folder1)
    x_train = np.load(folder1)
    
    print(f"Shape of x_train: {x_train.shape}")
    print(f"First few rows of x_train:\n{x_train[:5]}")
    # dataset_name = str(sys.argv[1])
    # seed = str(sys.argv[2])
    # folder1 =  '/nfs/home/wangpe/split_73' + '/' + 'train' + str(dataset_name) + ".npy"
    x_train = np.load(folder1)
    start = time.time()
    pop,min_fitness,unique_number = main(seed,x_train)# چند مجموعه ویژگی تولید میشه 
    end = time.time()
    running_time = end - start
    pop1 = delete_duplicate(pop)
    pop_fit = [ind.fitness.value for ind in pop1]
    EXA_array = np.array(pop1)
    saveFile.saveAllfeature2(seed, dataset_name, EXA_array)
    saveFile.saveAllfeature3(seed, dataset_name, pop_fit)
    saveFile.saveAllfeature5(seed, dataset_name, unique_number)
    saveFile.saveAllfeature6(seed, dataset_name, min_fitness)
    saveFile.saveAllfeature7(seed, dataset_name, running_time)
