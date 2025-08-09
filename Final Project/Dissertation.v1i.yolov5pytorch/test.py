#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 Ultralytics YOLOv8 進行訓練的範例腳本
請確保 data.yaml 與資料集的路徑對應正確
"""

from ultralytics import YOLO

def main():
    # 建立 YOLO 模型（使用預訓練權重 yolov8n.pt）
    model = YOLO("yolov8n.pt")
    
    # 開始訓練
    model.train(
        data="data.yaml",   # 指向同資料夾下的 data.yaml
        epochs=1,         # 訓練輪數，可依需求調整
        imgsz=640,          # 訓練圖片大小
        batch=16,           # 批次大小，可依硬體資源調整
        name="yolov8n_exp"  # 訓練結果輸出資料夾名稱
    )

if __name__ == "__main__":
    main()
