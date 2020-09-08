# sift_bow_libsvm


# bao-cao-thi-giac

****ĐỘ CHÍNH XÁC MÔ HÌNH (latest) : 82.647% *****

Thiết kế báo cáo bao gồm các file :

(các file sử dụng không cần quan tâm các file khác vì trong lúc tìm hiểu test code pla pla các thứ)

---baocaothigiac.ui : giao diện cho đồ án


---main_nhandang.py : load các model (svm,kmean,centers_words) từ main_training.py và sau đó predict ảnh mới


---main_training.py : trích sift + training Bow với MiniBatchKmean tối ưu về mặt thời gian hơn kmeans(sklearn) 
và BowTrainer(opencv) + training libsvm


---utils.py : các hàm thông dụng tai sử dụng nhiều lần . bao gồm đọc list folder, tập tin của folder nào đó
,và các hàm so các feature của 1 ảnh với cụm tính trước và cho ra 1 vector histogram


---các model được lưu lại với đuôi .pkl :
        
------------------centroids_words?.pkl: (? là là version để xem xét lại mô hình tối ưu)

------------------model?.pkl : (? là version các mô hình libSVM được lưu lại)

------------------mbk?.pkl : (? là version các mô hình kmeans được lưu lại)





****************************************************************************************************************************************

***ko liên quan đề tài : PCA face recongzie với eigenfaces:***

---Mỗi Ảnh đầu vào (M dòng N cột)-> SIFT -> ma trận descriptor(N dòng 128 cột) ->lấy descriptor đem tách thành phàn chính bằng PCA (bằng cách dùng thư viện hoặc tính SVD và dựa vào 3 ma trận sau khi tách SVD (UEV)  dựa vào E ma trận chéo để giữ thành phần quan trọng ) bước này sau khi PCA thì ra được 1 ma trận đã được nén không còn là Nx128 mà là N'x K với K<128 -> Flatten ma trận N' x  K ta được 1 vector có N' * K chiều (có thể là quá lớn số chiều ví dụ : N' =500 dòng , K còn 100 chiều thì là 1 vector 5000 chiều  chúng ta có thể giảm số chiều vector này bằng cách chọn lại số K ở bước sau khi PCA)

***cứ M ảnh theo tóm tắt trên thì ta sẽ có nhiều M cái vector N'*K chiều lúc này ta gán nhãn cho nó luôn***

----có 1 dataset có nhãn vậy dùng các máy học tương ứng KNN,svm....


