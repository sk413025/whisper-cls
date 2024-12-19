import os
import numpy as np
from joblib import load, dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import logging
from datetime import datetime

def setup_logger():
    # 創建 logs 資料夾
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 設定日誌檔名，包含時間戳記
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f'whisper_classifier_{timestamp}.log')
    
    # 設定 logging 格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file

def load_data():
    layers = [3, 4, 5]
    features_H = []  # 健康人的特徵
    features_S = []  # 病人的特徵
    
    # 讀取每一層的資料
    for layer in layers:
        # 讀取健康人資料
        h_path = f'whisper_layer_{layer}_H/whisper_layer_{layer}_H.pkl'
        h_data = load(open(h_path, 'rb'))
        logging.info(f"載入 {h_path}, 資料筆數: {len(h_data)}")
        
        # 讀取病人資料
        s_path = f'whisper_layer_{layer}_S/whisper_layer_{layer}_S.pkl'
        s_data = load(open(s_path, 'rb'))
        logging.info(f"載入 {s_path}, 資料筆數: {len(s_data)}")
        
        # 處理健康人資料
        for i, h in enumerate(h_data):
            # 如果是第一層，初始化特徵向量
            if layer == 3:
                features_H.append([])
            # 計算全局平均值作為特徵
            h_features = np.mean(h, axis=(0, 1))  # 將時間和特徵維度平均
            features_H[i].extend(h_features)
            
        # 處理病人資料
        for i, s in enumerate(s_data):
            # 如果是第一層，初始化特徵向量
            if layer == 3:
                features_S.append([])
            # 計算全局平均值作為特徵
            s_features = np.mean(s, axis=(0, 1))  # 將時間和特徵維度平均
            features_S[i].extend(s_features)
    
    # 轉換為 numpy array
    X_H = np.array(features_H)
    X_S = np.array(features_S)
    
    # 合併資料並創建標籤
    X = np.vstack([X_H, X_S])
    y = np.hstack([np.zeros(len(X_H)), np.ones(len(X_S))])  # 0表示健康，1表示病人
    
    logging.info(f"\n特徵維度: {X.shape}")
    logging.info(f"健康人數: {len(X_H)}, 病人數: {len(X_S)}")
    
    return X, y

def train_classifier(X, y):
    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 建立和訓練模型
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # 進行交叉驗證
    cv_scores = cross_val_score(clf, X, y, cv=5)
    logging.info(f"\n5-fold 交叉驗證分數: {cv_scores}")
    logging.info(f"平均準確率: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # 在測試集上評估
    y_pred = clf.predict(X_test)
    logging.info(f"\n分類報告:")
    logging.info(classification_report(y_test, y_pred, 
                                    target_names=['健康', '病人']))
    
    # 儲存模型
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, 'whisper_classifier.joblib')
    dump(clf, model_path)
    logging.info(f"\n模型已儲存至: {model_path}")
    
    return clf

def main():
    log_file = setup_logger()
    logging.info(f"開始訓練分類器，日誌檔案: {log_file}\n")
    
    try:
        # 載入資料
        X, y = load_data()
        
        # 訓練分類器
        clf = train_classifier(X, y)
        
        logging.info("\n訓練完成！")
        
    except Exception as e:
        logging.error(f"\n發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()
