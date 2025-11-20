import os
import shutil
import glob
import json
import yaml  # PyYAML 설치 필요: pip install pyyaml

# 파일 리스트 가져오기
training_file_pattern = 'Data/Json/Training/TL_딸기_병해충피해이미지/*.json'
validation_file_pattern = 'Data/Json/Validation/VL_딸기_병해충피해이미지/*.json'
training_files = glob.glob(training_file_pattern, recursive=True)
validation_files = glob.glob(validation_file_pattern, recursive=True)

# 경로 설정
train_image_src_dir = 'Data/Images/Training/TS_딸기_병해충피해이미지/'
train_image_dst_dir = 'Yolo/train/images'
train_label_dst_dir = 'Yolo/train/labels'
val_image_src_dir = 'Data/Images/Validation/VS_딸기_병해충피해이미지/'
val_image_dst_dir = 'Yolo/val/images'
val_label_dst_dir = 'Yolo/val/labels'

os.makedirs(train_image_dst_dir, exist_ok=True)
os.makedirs(train_label_dst_dir, exist_ok=True)
os.makedirs(val_image_dst_dir, exist_ok=True)
os.makedirs(val_label_dst_dir, exist_ok=True)

# 바이너리 클래스: 0=정상, 1=이상
def get_binary_class_id(class_name):
    if class_name in ['잿빛곰팡이병', '흰가루병']:
        return 1  # 이상 (병)
    else:
        return 0  # 정상

def to_yolo_format(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    return x_center, y_center, width, height

def process_files(file_list, image_src_dir, image_dst_dir, label_dst_dir):
    for json_file in file_list:
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        image = data['description']['image']
        img_width = data['description']['width']
        img_height = data['description']['height']

        extracted = {item['name']: item['value'] for item in data.get('metadata', [])}
        class_name = extracted.get('작물상태코드', '').lstrip('\ufeff')
        class_id = get_binary_class_id(class_name)

        bbox_list = []
        for item in data.get('result', []):
            if item.get('type') == 'bbox':
                bbox = {
                    'x': item.get('x'),
                    'y': item.get('y'),
                    'w': item.get('w'),
                    'h': item.get('h')
                }
                bbox_list.append(bbox)

        src_image_path = os.path.join(image_src_dir, image)
        dst_image_path = os.path.join(image_dst_dir, image)

        if not os.path.isfile(src_image_path):
            print(f"Warning: Image file not found, skipping: {src_image_path}")
            continue

        shutil.copy2(src_image_path, dst_image_path)

        image_basename = os.path.splitext(image)[0]
        label_path = os.path.join(label_dst_dir, image_basename + '.txt')

        with open(label_path, 'w', encoding='utf-8') as f:
            for bbox in bbox_list:
                x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                x_c, y_c, w_norm, h_norm = to_yolo_format(x, y, w, h, img_width, img_height)
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        print(f"Processed {image} and saved annotation.")

# 학습/검증 데이터 처리
process_files(training_files, train_image_src_dir, train_image_dst_dir, train_label_dst_dir)
process_files(validation_files, val_image_src_dir, val_image_dst_dir, val_label_dst_dir)

# data.yaml 생성
data_yaml_path = 'Yolo/data.yaml'
data_yaml = {
    'train': 'train/images',
    'val': 'val/images',
    'nc': 2,
    'names': ['정상', '이상']
}

with open(data_yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(data_yaml, f, allow_unicode=True)

print(f"data.yaml 생성 완료: {data_yaml_path}")
