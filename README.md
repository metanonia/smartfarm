### AI-HUB(https://aihub.or.kr/) 지능형 스마트팜(오이, 딸기) 데이터 이용


#### Prepare Dectection
```python make_yolo.py```

#### Training Detection
```yolo train model=yolo11n.pt data=Yolo/data.yaml epochs=100 imgsz=320 task=detect```<br><br>
```yolo train model=yolo11n.pt data=Yolo/data.yaml epochs=200 imgsz=320 batch=16 mosaic=1.0 mixup=0.1 degrees=15 flipud=0.5 fliplr=0.5  scale=0.9 translate=0.1 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 iou=0.5 lr0=0.001 patience=100 optimizer=AdamW```

#### Prepare Classification
```python make_yolo_classification.py```

#### Training Classification
```yolo classify train data=Yolo_Classification model=yolo11n-cls.pt epochs=100 imgsz=224```

### Test (Dection + Classification)
```python detect_and_classification.py```

### [참고]
#### Dectection With Class
```python make_yolo_part_class.py```