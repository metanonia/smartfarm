### AI-HUB(https://aihub.or.kr/) 
#### 지능형 스마트팜(딸기, 오이) 테이터 중 딸기
#### 시설작물 개체 이미지 및 시설작물 질병 이미지 데이터 중 딸기
#### 시설 작물 질병 진단 이미지 중 딸기
  - 질병코드: 딸기잿빛곰팡이병(07) 딸기흰가루병(08)
  - 쵤영부위: None, Fruit, Flower, Leaf, Branch, Stem, Root

<br>

#### Prepare Dectection (지능형 스마트팜 및 시설작물 질병 이미지 데이터 이용)
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
```yolo val  model=./best.pt data=Yolo/data.yaml imgsz=320 save=True save_txt=True save_conf=True`

#### Prepare Classification
```python make_yolo_classification.py```

#### Training Classification
```yolo classify train data=Yolo_Classification model=yolo11n-cls.pt epochs=100 imgsz=224```

### Test (Dection + Classification)
```python detect_and_classification.py```
