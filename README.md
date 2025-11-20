### AI-HUB(https://aihub.or.kr/) 지능형 스마트팜(오이, 딸기) 데이터 이용


#### Prepare Dectection
```python make_yolo.py```

### Detection
```yolo train model=yolo11n.pt data=Yolo/data.yaml epochs=100 imgsz=320 task=detect name=yolo11n_custom```

### Prepare Classification

### Classification


### Dectection With Class
```python make_yolo_part_class.py```