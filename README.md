### AI-HUB(https://aihub.or.kr/) 지능형 스마트팜(오이, 딸기) 데이터 이용


#### Prepare Dectection
```python make_yolo.py```

#### Training Detection
```yolo train model=yolo11n.pt data=Yolo/data.yaml epochs=100 imgsz=320 task=detect name=yolo11n_custom```

#### Prepare Classification
```python make_yolo_classification.py```

#### Training Classification
```yolo classify train data=Yolo_Classification model=yolo11n-cls.pt epochs=100 imgsz=224```

### Test (Dection + Classification)


### [참고]
#### Dectection With Class
```python make_yolo_part_class.py```