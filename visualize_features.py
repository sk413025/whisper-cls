import os
import numpy as np
from joblib import load
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # 設定後端為 Agg
import matplotlib.pyplot as plt
import logging
from datetime import datetime

def setup_logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f'whisper_viz_{timestamp}.log')
    
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
        h_path = os.path.join('data', f'whisper_layer_{layer}_H/whisper_layer_{layer}_H.pkl')
        h_data = load(open(h_path, 'rb'))
        logging.info(f"載入 {h_path}, 資料筆數: {len(h_data)}")
        
        # 讀取病人資料
        s_path = os.path.join('data', f'whisper_layer_{layer}_S/whisper_layer_{layer}_S.pkl')
        s_data = load(open(s_path, 'rb'))
        logging.info(f"載入 {s_path}, 資料筆數: {len(s_data)}")
        
        # 處理健康人資料
        for i, h in enumerate(h_data):
            if layer == 3:
                features_H.append([])
            h_features = np.mean(h, axis=(0, 1))  # 將時間和特徵維度平均
            features_H[i].extend(h_features)
            
        # 處理病人資料
        for i, s in enumerate(s_data):
            if layer == 3:
                features_S.append([])
            s_features = np.mean(s, axis=(0, 1))  # 將時間和特徵維度平均
            features_S[i].extend(s_features)
    
    X_H = np.array(features_H)
    X_S = np.array(features_S)
    
    logging.info(f"\n特徵維度:")
    logging.info(f"健康組: {X_H.shape}")
    logging.info(f"病人組: {X_S.shape}")
    
    return X_H, X_S

def visualize_features(X_H, X_S):
    # 合併資料
    X = np.vstack([X_H, X_S])
    
    try:
        clf = load('models/whisper_classifier.joblib')
        # 獲取特徵權重
        feature_weights = np.abs(clf.coef_[0])
        
        # 分析每一層的權重分布
        features_per_layer = len(feature_weights) // 3
        for i in range(3):
            layer_weights = feature_weights[i*features_per_layer:(i+1)*features_per_layer]
            layer_importance = np.mean(layer_weights)
            logging.info(f"\n第 {i+3} 層平均權重: {layer_importance:.4f}")
            logging.info(f"第 {i+3} 層權重標準差: {np.std(layer_weights):.4f}")
            logging.info(f"第 {i+3} 層最大權重: {np.max(layer_weights):.4f}")
            
        # 找出最重要的特徵
        top_k = 100  # 選擇前100個最重要的特徵
        top_indices = np.argsort(feature_weights)[-top_k:]
        logging.info(f"\n最重要的 {top_k} 個特徵所在層:")
        for idx in top_indices:
            layer_idx = idx // features_per_layer + 3
            feature_idx = idx % features_per_layer
            logging.info(f"特徵 {idx}: 層 {layer_idx}, 位置 {feature_idx}, 權重 {feature_weights[idx]:.4f}")
        
        # 只保留重要特徵
        importance_threshold = np.percentile(feature_weights, 75)  # 保留前25%的特徵
        important_features = feature_weights > importance_threshold
        logging.info(f"\n保留特徵數量: {np.sum(important_features)}/{len(feature_weights)}")
        
        # 加權特徵，並只使用重要特徵
        X_weighted = X[:, important_features] * feature_weights[important_features]
        
        # 標準化特徵
        X_weighted = (X_weighted - np.mean(X_weighted, axis=0)) / np.std(X_weighted, axis=0)
        
        logging.info("\n使用篩選後的重要特徵進行加權")
    except Exception as e:
        X_weighted = X
        logging.info(f"\n載入分類器時發生錯誤: {str(e)}")
        logging.info("使用原始特徵")
    
    # 創建圖片保存目錄
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # t-SNE 降維，調整參數
    logging.info("\n執行 t-SNE 降維...")
    tsne_params = {
        'n_components': 2,
        'perplexity': 15,  # 調小perplexity以更注重局部結構
        'early_exaggeration': 20,  # 增加early_exaggeration以更好地分離群集
        'learning_rate': 200,  # 增加learning_rate
        'n_iter': 2000,  # 增加迭代次數
        'random_state': 42
    }
    logging.info(f"t-SNE 參數: {tsne_params}")
    
    tsne = TSNE(**tsne_params)
    X_tsne = tsne.fit_transform(X_weighted)
    
    # 計算群內和群間距離
    H_tsne = X_tsne[:len(X_H)]
    S_tsne = X_tsne[len(X_H):]
    
    H_center = np.mean(H_tsne, axis=0)
    S_center = np.mean(S_tsne, axis=0)
    
    H_intra_dist = np.mean([np.linalg.norm(x - H_center) for x in H_tsne])
    S_intra_dist = np.mean([np.linalg.norm(x - S_center) for x in S_tsne])
    inter_dist = np.linalg.norm(H_center - S_center)
    
    logging.info(f"\n健康組內平均距離: {H_intra_dist:.4f}")
    logging.info(f"病人組內平均距離: {S_intra_dist:.4f}")
    logging.info(f"組間距離: {inter_dist:.4f}")
    logging.info(f"分離度 (組間/組內): {inter_dist/((H_intra_dist + S_intra_dist)/2):.4f}")
    
    # 繪製 t-SNE 結果
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X_tsne[:len(X_H), 0], X_tsne[:len(X_H), 1], 
              c='blue', marker='o', label='Healthy', alpha=0.7)
    ax.scatter(X_tsne[len(X_H):, 0], X_tsne[len(X_H):, 1], 
              c='red', marker='^', label='Patient', alpha=0.7)
    
    # 繪製群中心
    ax.scatter(H_center[0], H_center[1], c='blue', marker='*', s=200, label='Healthy Center')
    ax.scatter(S_center[0], S_center[1], c='red', marker='*', s=200, label='Patient Center')
    
    ax.set_title('Weighted t-SNE Visualization (Important Features)')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    tsne_path = os.path.join('plots', 'whisper_weighted_tsne_v3.png')
    fig.savefig(tsne_path, dpi=300, bbox_inches='tight')
    logging.info(f"\n改進後的加權 t-SNE 圖已保存至: {tsne_path}")
    plt.close(fig)

    # 繪製權重分布圖
    if 'feature_weights' in locals():
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(feature_weights, alpha=0.7)
        ax.set_title('Feature Importance Distribution')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Normalized Weight')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        weights_path = os.path.join('plots', 'feature_weights_v3.png')
        fig.savefig(weights_path, dpi=300, bbox_inches='tight')
        logging.info(f"特徵權重分布圖已保存至: {weights_path}")
        plt.close(fig)

def main():
    log_file = setup_logger()
    logging.info(f"開始特徵視覺化，日誌檔案: {log_file}\n")
    
    try:
        # 載入資料
        X_H, X_S = load_data()
        
        # 視覺化特徵
        visualize_features(X_H, X_S)
        
        logging.info("\n視覺化完成！")
        
    except Exception as e:
        logging.error(f"\n發生錯誤: {str(e)}")

if __name__ == "__main__":
    main() 