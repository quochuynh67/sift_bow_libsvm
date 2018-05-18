import sys
from svmutil import *
import cv2
import numpy as np
from utils import *
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMessageBox

class BaoCaoThiGiac(QDialog):
    def __init__(self):
        super().__init__()
        loadUi('baocaothigiac.ui', self)

        self.image=None
        self.imageTest=None
        self.dict_classes=None
        self.dict_pokemon=None
        self.centers_word=None
        self.modelSVM=None
        self.modelKMean=None

        self.setGeometry(self.frameSize().width() // 4
                         , self.frameSize().height() // 4
                         , 800
                         , 400
                         )
        self.btnBrowser.clicked.connect(self.LoadHinhAnh)
        self.btnNhanDang.clicked.connect(self.Predict)
        self.LoadModel()

    def LoadModel(self):

        self.dict_classes = {
            1.0: 'Máy bay',
            2.0: 'Bon Sai',
            3.0: 'Rùa biển',
            4.0: 'Thuyền Bườm',
            5.0: 'Đồng Hồ'}

        # load centers words
        self.centers_word = joblib.load("./centers_word1.pkl")
        # load model svm
        self.modelSVM = svm_load_model('./model1.pkl')
        # load model kmean
        self.modelKMean = joblib.load("./mbk1.pkl")

    def LoadHinhAnh(self):
        fname = QFileDialog.getOpenFileName(self,None,'./img_test/')
        self.edtBrowser.setText(fname[0])
        self.txtResult.setText("")
        self.image=cv2.imread(str(fname[0]).replace("/","\\\\"))
        self.imageTest=cv2.imread(str(fname[0]).replace("/","\\\\"))
        self.image=cv2.resize(self.image,
                              (self.image.shape[1]*2 if self.image.shape[1] <500 else self.image.shape[1]//2
                               ,self.image.shape[0]+70 if self.image.shape[0] <500 else self.image.shape[0]//2)
                              )
        self.DisplayImage()

    def Predict(self):
        if self.image != None:
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(self.imageTest, None)

            #####cach 1
            # hist_new_image = getVectorHistogram(self.centers_word, descriptors, 999)

            #####cach2
            # hist_new_image=np.zeros(k)
            # for x,des in enumerate(descriptors):
            #     idx = self.modelKMean.predict([des])
            #     hist_new_image[idx]+=1
            # print(hist_new_image.tolist())

            print(self.modelKMean)
            ####cach 3
            hist_new_image=np.bincount(self.modelKMean.predict(descriptors), minlength=self.modelKMean.n_clusters)
            print(hist_new_image.tolist())

            x0, max_idx = gen_svm_nodearray(hist_new_image.tolist())
            label = libsvm.svm_predict(self.modelSVM, x0)
            print(label)
            print(self.dict_classes[label])
            self.txtResult.setText(self.dict_classes[label])
        else:
            QMessageBox.about(self,"Thông báo!", "Vui lòng chọn ảnh cần nhận dạng!")


    def DisplayImage(self):
        self.image = QImage(self.image
                            , self.image.shape[1]
                            , self.image.shape[0]
                            , self.image.strides[0]
                            , QImage.Format_RGB888)
        self.image = self.image.rgbSwapped()
        self.txtShowImage.setPixmap(QPixmap.fromImage(self.image))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BaoCaoThiGiac()
    window.setWindowTitle('Nhận dạng ảnh (SIFT + BOW + libSVM)')
    window.show()
    sys.exit(app.exec_())







#80% , K=900 ,5 class, 1500 anh, 1500 instance
# 80.8153% (674/834) (classification) 1800k , batch45
#Accuracy = 81.295% (678/834) (classification) 3000K batch 30