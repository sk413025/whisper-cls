import os
import numpy as np
from joblib import load
import logging
from datetime import datetime

def setup_logger():
    # 創建 logs 資料夾（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 設定日誌檔案名稱，包含時間戳記
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f'whisper_check_{timestamp}.log')
    
    # 設定 logging 格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同時輸出到控制台
        ]
    )
    return log_file

def check_whisper_layers():
    log_file = setup_logger()
    logging.info(f"開始檢查 Whisper 層輸出，日誌檔案: {log_file}\n")
    
    # 定義要檢查的層數和狀態
    layers = [3, 4, 5]
    states = ['H', 'S']
    
    # 遍歷所有資料夾
    for layer in layers:
        for state in states:
            folder_name = f'whisper_layer_{layer}_{state}'
            
            # 確認資料夾存在
            if not os.path.exists(folder_name):
                logging.info(f"找不到資料夾: {folder_name}")
                continue
                
            # 列出資料夾中的所有檔案
            files = os.listdir(folder_name)
            
            logging.info(f"\n檢查 {folder_name}:")
            for file in files:
                file_path = os.path.join(folder_name, file)
                
                if file.endswith('.npy'):
                    # 讀取 .npy 檔案
                    data = np.load(file_path)
                    logging.info(f"檔案: {file}, Shape: {data.shape}")
                
                elif file.endswith('.pkl'):
                    # 使用 joblib 讀取 .pkl 檔案
                    try:
                        data = load(file_path)
                        if isinstance(data, np.ndarray):
                            logging.info(f"檔案: {file}, Shape: {data.shape}, 類型: {data.dtype}")
                        else:
                            msg = f"檔案: {file}, 類型: {type(data)}"
                            if hasattr(data, '__len__'):
                                msg += f", 長度: {len(data)}"
                            logging.info(msg)
                    except Exception as e:
                        logging.error(f"檔案: {file}, 讀取錯誤: {str(e)}")
    
    logging.info("\n檢查完成！")

if __name__ == "__main__":
    check_whisper_layers() 