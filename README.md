# Whisper 中間層特徵分析

這個專案使用 Whisper 模型的中間層輸出來分析健康人和病人的語音特徵差異。

## 專案結構

```
whisper-cls/
├── check_whisper_outputs.py # 檢查 Whisper 中間層輸出
├── train_whisper_classifier.py # 訓練分類器
├── visualize_features.py # 特徵視覺化
├── logs/ # 日誌檔案
├── models/ # 儲存訓練好的模型
├── plots/ # 視覺化圖表
└── whisper_layer_{3,4,5}{H,S}/ # Whisper 中間層特徵
└── whisper_layer_{3,4,5}{H,S}.pkl

```

## 執行流程

1. 檢查特徵資料：

```bash
python check_whisper_outputs.py
```

2. 訓練分類器：

```bash
python train_whisper_classifier.py
```

使用 Logistic Regression 訓練一個二元分類器，區分健康人和病人的語音特徵。
- 輸出：models/whisper_classifier.joblib

3. 視覺化特徵：

```bash
python visualize_features.py
```

將高維特徵使用 t-SNE 降維視覺化，並分析特徵的分布情況。
- 輸出：plots/whisper_weighted_tsne_v3.png
- 輸出：plots/feature_weights_v3.png

## 資料說明

- whisper_layer_{3,4,5}_H：健康人的中間層特徵
- whisper_layer_{3,4,5}_S：病人的中間層特徵
- 每層特徵維度：512
- 總特徵維度：1536 (512 x 3)

將高維特徵使用 t-SNE 降維視覺化，並分析特徵的分布情況。
- 輸出：plots/whisper_weighted_tsne_v3.png
- 輸出：plots/feature_weights_v3.png

## 資料說明

- whisper_layer_{3,4,5}_H：健康人的中間層特徵
- whisper_layer_{3,4,5}_S：病人的中間層特徵
- 每層特徵維度：512
- 總特徵維度：1536 (512 x 3)

將高維特徵使用 t-SNE 降維視覺化，並分析特徵的分布情況。
- 輸出：plots/whisper_weighted_tsne_v3.png
- 輸出：plots/feature_weights_v3.png

## 資料說明

- whisper_layer_{3,4,5}_H：健康人的中間層特徵
- whisper_layer_{3,4,5}_S：病人的中間層特徵
- 每層特徵維度：512
- 總特徵維度：1536 (512 x 3)