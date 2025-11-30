## AI-HUB(https://aihub.or.kr/) 
### 학습데이터
#### - 지능형 스마트팜(딸기, 오이) 테이터 중 딸기 (AIHUB)
#### - 시설작물 개체 이미지 및 시설작물 질병 이미지 데이터 중 딸기 (AIHUB)
### 검증데이터
#### - 시설 작물 질병 진단 이미지 중 딸기 (AIHUB)
  - 질병코드: 딸기잿빛곰팡이병(07) 딸기흰가루병(08)
  - 쵤영부위: None, Fruit(1), Flower(2), Leaf(3), Branch(4), Stem(5), Root(6)
#### - PlantDoc Classification dataset (https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset)
#### - Strawberry Disease Detection (https://universe.roboflow.com/strawberry-disease/strawberry-disease-detection-dataset/dataset/4)


### Prepare Dectection (지능형 스마트팜 및 시설작물 질병 이미지 데이터 이용)
```python make_yolo.py```<br>
```python make_yolo_part_class.py```<br>
- json의 이미지크기와 실제 데이터 이미지 크기가 다름
- 단일 객체에 복수의 바운딩 박스가 있어서 ReCall 값이 낮게 나옴 (단일 바운딩 박스로 처리)

### Training Detection (yolo11n)
```yolo train model=yolo11n.pt data=Yolo/data.yaml epochs=100 imgsz=640 task=detect```<br>
```yolo train model=best.pt data=Yolo/data.yaml epochs=100 imgsz=640 multi_scale=True mosaic=10.0 mixup=0.1 task=detect```<br>
```yolo train model=yolo11n.pt data=Yolo/data.yaml epochs=100 imgsz=640 batch=16 mosaic=0.5 mixup=0.02 degrees=20 ```<br>
- 작은 사이즈로 학습시, 큰 이미지를 이용하여 병증을 detection하는 경우, 검출이 잘 안됨

### IoU값 변경 Validation
```yolo val  model=./best.pt data=Yolo/data.yaml imgsz=640 iou=0.3```

### False Negative 확인
```yolo val  model=./best.pt data=Yolo/data.yaml imgsz=320 save=True save_txt=True save_conf=True`

### Prepare Classification
```python make_yolo_classification.py```

### Training Classification
```yolo classify train data=Yolo_Classification model=yolo11n-cls.pt epochs=100 imgsz=224```

### Test (Dection + Classification)
```python detect_and_classification.py```

### ONNX 생성
```yolo export model=detect_model.pt format=onnx opset=17```
