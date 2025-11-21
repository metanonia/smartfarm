### AI-HUB(https://aihub.or.kr/) 지능형 스마트팜(오이, 딸기) 데이터 이용
<br>
### 딸기 병해충피해이미지로 검출 학습했을 때.. ReCall 값이 너무 낮게 나옴<br>
<br>

#### Prepare Dectection
```python make_yolo.py```

#### Training Detection
```yolo train model=yolo11n.pt data=Yolo/data.yaml epochs=100 imgsz=320 task=detect```<br><br>
```yolo train model=yolo11n.pt data=Yolo/data.yaml epochs=200 imgsz=320 batch=16 mosaic=0.5 mixup=0.02 degrees=10-20 ```

#### Prepare Classification
```python make_yolo_classification.py```

#### Training Classification
```yolo classify train data=Yolo_Classification model=yolo11n-cls.pt epochs=100 imgsz=224```

### Test (Dection + Classification)
```python detect_and_classification.py```

### [참고]
#### Dectection With Class
```python make_yolo_part_class.py```
#### 열매_잿빛곰팡이병, 잎_흰가루병 와 그 외로 구분
```python make_yolo_binary.py```