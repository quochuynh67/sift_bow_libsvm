import os
import cv2
import numpy as np

static_path ='./images_train_abcd'
k=3000
def getDanhSachFolderName():
    return os.listdir(static_path)



def getDanhSachFiles(f_n):
    return os.listdir(static_path+"/"+f_n)


def getListAbsolutePath_ofFolder(folder_name):
    list_path=[]
    files=getDanhSachFiles(folder_name)
    for x in range(len(files)):
        list_path.append(static_path+"/"+folder_name+"/"+files[x])
    return list_path

# try:
#     print(getListAbsolutePath_ofFolder('_rooster'))
# except:
#     print("ten folder bi sai!")
def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

'''
@doi so 1 : cac vector cum
@doi so 2: cac feature cua 1 anh
@doi so 3 : index class 
'''
def getVectorHistogram(centroid,descriptors,class_index):
    #len(centroid) + 1 la do them 1 cot cho label
    hist_cluster=np.zeros((1,len(centroid)+1))#hist cua 1 anh ...tinh gia tri cua moi word bang cach dem so feature trong cum
    for i in range(len(descriptors)):#duyet qua cac feature tinh khoang cach feature nay voi 10 cum no gan cum nao thi tang index cum do len
        temp = []
        for j in range(len(centroid)):
            temp.append(distance(descriptors[i], centroid[j]))
        hist_cluster[0,np.argmin(temp)] += 1#lay index min trong temp boi vi temp chua cac khoang cach
        # hist_cluster[0,np.argmin(temp)+1] += 1#lay index min trong temp boi vi temp chua cac khoang cach
        hist_cluster[0,-1] = class_index#cot cuoi la label
        # hist_cluster[0,0] = class_index#cot dau la label
    return hist_cluster

def getVectorVersion2(descriptor,class_name,model,K):
    histo = np.zeros(K + 1)
    for x, des in enumerate(descriptor):
        idx = model.predict([des])
        histo[idx] += 1
        histo[-1] = class_name
    return histo

