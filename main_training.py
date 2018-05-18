from svmutil import *
import cv2
import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans


K = k

#load file name
try:
    list_path_airplanes     = getListAbsolutePath_ofFolder('airplanes')
    list_path_bonsai    = getListAbsolutePath_ofFolder('bonsai')
    list_path_turtle    = getListAbsolutePath_ofFolder('hawksbill')
    list_path_ketch = getListAbsolutePath_ofFolder('ketch')
    list_path_watch   = getListAbsolutePath_ofFolder('watch')

    soluong_img =len(list_path_airplanes)+len(list_path_bonsai) +len(list_path_turtle)+len(list_path_ketch)+len(list_path_watch)
    print("Tong so anh =", str(soluong_img))
except:
    print("Kiem tra lai folder name")



#hist cac image ,, images train de train svm
all_sift_descriptors_extend=[]
all_sift_descriptors_append=[]
def trichXuatsift(path,class_name,class_index):
    for i,path_i in enumerate(path):
        # print("[", class_name, "] Trich xuat dac trung sift image  " + str(i + 1))
        img = cv2.imread(path[i], cv2.IMREAD_GRAYSCALE)
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        # print("[",class_name ,"] so luong features = " , len(descriptors))
        all_sift_descriptors_extend.extend(descriptors)
        '''all_sift_descriptor_appen luu 1 list 2 gia tri. trong luc trich sift khoi phai trich sift lan 2
                -gia tri 1:descriptors
                -gia tri 2: class'''
        all_sift_descriptors_append.append([descriptors,class_index])

print("Bat dau trich xuat SIFT...")

trichXuatsift(list_path_airplanes   ,"AIRPLANES",1)
trichXuatsift(   list_path_bonsai      ,"BONSAI",2)
trichXuatsift(   list_path_turtle      ,"TURTLE",3)
trichXuatsift(    list_path_ketch       ,"KETCH",4)
trichXuatsift(    list_path_watch       ,"WATCH",5)


all_sift_descriptors_extend=np.asarray(all_sift_descriptors_extend)
print("So luong features = " , len(all_sift_descriptors_extend))
print("Trich xuat SIFT xong!")

print("Training bag of words...")
mbk = MiniBatchKMeans(init='k-means++', n_clusters=K, batch_size=30,
                      max_no_improvement=10, verbose=0,init_size=3*K)
mbk.fit(all_sift_descriptors_extend)

centers_word=mbk.cluster_centers_


# BoWs = cv2.BOWKMeansTrainer(K)
# BoWs.add(all_sift_descriptors_extend)
# centers_word = BoWs.cluster()
# print('centers_word' ,centers_word)

joblib.dump(centers_word, "centers_word.pkl") #save file centers
joblib.dump(mbk, "mbk.pkl") #save file centers
print("Training bag of words xong!")
print(" len append = " ,str(len(all_sift_descriptors_append)))

#images train
print("Chuan bi images train histogram ...")
all_histogram=[]

for i,descriptor_i in enumerate(all_sift_descriptors_append):

    ###cach 1
    # all_histogram.extend(getVectorHistogram(centers_word,descriptor_i[0],descriptor_i[1]))

    ###cach 2
    # histo = np.zeros(K+1)
    # for x,des in enumerate(descriptor_i[0]):
    #     # print(des)
    #     idx = mbk.predict([des])
    #     histo[idx] += 1
    #     histo[-1]=descriptor_i[1]
    # all_histogram.append(histo)

    ###cach 3
    histo=np.bincount(mbk.predict(descriptor_i[0]), minlength=mbk.n_clusters)
    histo[-1]=descriptor_i[1]
    all_histogram.append(histo)


all_histogram=np.asarray(all_histogram).tolist()
print("do dai images train = " , len(all_histogram), " moi hist co so chieu =" ,len(all_histogram[0]))
print("images train : " , all_histogram[-1])




#chuyen doi dinh dang images train phu hop libsvm
y=[]
x=[]


'''
khong theo format libsvm
x x x x x x class
'''
for i,hist_i in enumerate(all_histogram):
    y.append(hist_i[-1])
    x.append(hist_i[:-1])




'''
theo format class index:value
'''
# for i,hist_i in enumerate(all_histogram):
#     temp={}
#     y.append(hist_i[-1])
#     for j,each_in_hist_i in enumerate(hist_i[:-1]):
#        if each_in_hist_i != 0 :
#            temp.update({j: each_in_hist_i})
#     x.append(temp)





# print("do dai label = ",len(y))
# print("do dai hist = ",len(x))
# print(y)
# print(x)
print("Chuan bi images train xong!")
#training libsvm
print("Training multi class voi svm...")




####linearSVC
# clf = svm.LinearSVC(random_state=1)
# X=np.array(x).tolist()
# y=np.array(y).tolist()
# size_test = 1 - (2/3.0)
# data_train, data_test, label_train, label_test=train_test_split(X, y,test_size=size_test,random_state=1)
# print("data_train :\n",data_train )
# print("data_test :\n",data_test )
# print("label_train :\n",label_train )
# print("label_test :\n",label_test )
# clf.fit(data_train, label_train)
# res =clf.predict(data_test)
#
# count = 0
# for i in range(len(res)):
# 	if res[i] == label_test[i]:
# 		count += 1
# joblib.dump(clf,'clf.pkl')
#
# print("Training multi class voi svm xong")
# print("Do chinh xac: {} %".format(count/len(label_test) * 100))




#####Libsvm
size_test = 1 - (2/3.0)
data_train, data_test, label_train, label_test=train_test_split(x, y,test_size=0.25,random_state=1)
print("data = " , len(x) ,  " ,so luong data train = " , len(data_train) , " ,so luong data test = ", len(data_test) )
print("data_train :\n",data_train )
print("data_test :\n",data_test )
print("label_train :\n",label_train )
print("label_test :\n",label_test )


problem  = svm_problem(label_train, data_train)
param = svm_parameter('-s 0 -t 0 -c 100')
model = svm_train(problem, param)
predict_labels, predict_acc, predict_val = svm_predict(label_test,data_test,model)
svm_save_model('model.pkl', model)


#100 c, k=3000,batchsize30 81%
#0.01 c,K=3000, batchsize30 , 99%
#90% ,K=3000, c=0.01 bszie30