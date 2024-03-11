# 資料來源:
    數據資料來源為美國加利福尼亞大學爾灣分校UCI的開源資料https://archive.ics.uci.edu/裡的
    https://archive.ics.uci.edu/dataset/161/mammographic+mass 乳房X光數值


# 研究動機:
    長年以來，癌症一直是台灣死因中最主要的原因之一，與第二名相比，其發生率高出了2倍。全球科學家一直在努力研究，希望能夠減少癌症所造成的悲劇。
    根據台灣衛福部國健署的統計數據，女性乳癌的發病率逐年上升，從110年的2913人佔比17.1%，提升至111年的24.1%。這樣的增長速度非常令人擔憂。
    乳癌檢測的主要方法之一是X光攝影檢查，其優點在於對早期乳癌的鈣化點具有敏感檢測能力，可以在疾病早期進行預防和治療。
    然而，這種檢查需要對乳房進行緊壓，可能會引起疼痛和低劑量輻射，因此通常建議每1到2年進行一次檢查。
    由於檢測的準確性無法達到100%，因此我們希望通過應用AI機器學習技術來提高檢測準確率，並準確判斷腫瘤的性質，是良性還是惡性。

# 大致步驟:
    1.先處理資料中含有缺失值或是極端值等數據不平等的情況再做標準化
    2.再使用各種機器學習中的模型來訓練和測試資料
      Decision Trees、RandomForestClassifier、SVM、KNN、Naive Bayes、Logistic Regression
    3.判斷哪個更具有效益且更高的準確度

# 詳細流程:
## 1.對資料進行基本的前處理、標準化: 
<img src="https://github.com/AlanChou0330/Breast-Cancer---machine-learning/blob/main/picture/data1.png" width="550"/>
<img src="https://github.com/AlanChou0330/Breast-Cancer---machine-learning/blob/main/picture/data2.png" width="400"/>

### 在繪製出直方圖和箱線圖來更直觀的看到資料前處理以及標準化帶來的變化:
##### 直方圖
<img src="https://github.com/AlanChou0330/Breast-Cancer---machine-learning/blob/main/picture/DataPreprocessing_1.png" width="600"/>
<img src="https://github.com/AlanChou0330/Breast-Cancer---machine-learning/blob/main/picture/DataPreprocessing_2.png" width="600"/>

##### 箱線圖
<img src="https://github.com/AlanChou0330/Breast-Cancer---machine-learning/blob/main/picture/DataPreprocessing_3.png" width="600"/>
<img src="https://github.com/AlanChou0330/Breast-Cancer---machine-learning/blob/main/picture/DataPreprocessing_4.png" width="600"/>

## 2.用決策樹來訓練測試數據集:
<img src="https://github.com/AlanChou0330/Breast-Cancer---machine-learning/blob/main/picture/Decision Trees.png" width="900"/>
<img src="https://github.com/AlanChou0330/Breast-Cancer---machine-learning/blob/main/picture/cv_DecisionTrees.png" width="600"/>

## 3.用隨機森林來訓練測試數據集:
<img src="https://github.com/AlanChou0330/Breast-Cancer---machine-learning/blob/main/picture/cv_RandomForest.png" width="600"/>

## 4.用SVM支撐向量機來訓練測試數據集:
<img src="https://github.com/AlanChou0330/Breast-Cancer---machine-learning/blob/main/picture/cv_SVM.png" width="600"/>

## 5.用KNN來訓練測試數據集:
<img src="https://github.com/AlanChou0330/Breast-Cancer---machine-learning/blob/main/picture/cv_KNN.png" width="600"/>

## 6.用Naive Bayes貝氏分類器來訓練測試數據集:
<img src="https://github.com/AlanChou0330/Breast-Cancer---machine-learning/blob/main/picture/cv_NaiveBayes.png" width="600"/>

## 7.用Logistic Regression邏輯迴歸來訓練測試數據集:
<img src="https://github.com/AlanChou0330/Breast-Cancer---machine-learning/blob/main/picture/cv_LogisticRegression.png" width="600"/>
