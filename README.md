### AI-HUB(https://aihub.or.kr/) 
#### 지능형 스마트팜(오이, 딸기) 와 시설작물(딸기) 개체 이미지 및 시설작물(딸기) 질병 이미지 데이터 이용
- 딸기: 지능형 스타트팜 (열매, 잎), 시설작물(잎)
<br>

#### Prepare Dectection
```python make_yolo.py```<br>
```python make_yolo_part_class.py```<br>
- json의 이미지크기와 실제 데이터 이미지 크기가 다름
- 단일 객체에 복수의 바운딩 박스가 있어서 ReCall 값이 낮게 나옴 (단일 바운딩 박스로 처리)

#### Training Detection
```yolo train model=yolo11n.pt data=Yolo/data.yaml epochs=100 imgsz=320 task=detect```<br><br>
```yolo train model=yolo11n.pt data=Yolo/data.yaml epochs=100 imgsz=320 batch=16 mosaic=0.5 mixup=0.02 degrees=20 ```

#### IoU값 변경 Validation
```yolo val  model=./best.pt data=Yolo/data.yaml imgsz=320 iou=0.3```

#### False Negative 확인
```yolo val  model=./best.pt data=Yolo/data.yaml imgsz=320 save=True save_txt=True save_conf=True```

#### Prepare Classification
```python make_yolo_classification.py```

#### Training Classification
```yolo classify train data=Yolo_Classification model=yolo11n-cls.pt epochs=100 imgsz=224```

### Test (Dection + Classification)
```python detect_and_classification.py```
