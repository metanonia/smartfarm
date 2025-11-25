import os
import shutil
import glob
import json
import yaml
import cv2


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

# 클래스 이름 -> ID 매핑
class_name_to_id = {}

def get_or_create_class_id(class_name: str) -> int:
    """질병명을 그대로 클래스 이름으로 사용."""
    if class_name not in class_name_to_id:
        class_name_to_id[class_name] = len(class_name_to_id)
    return class_name_to_id[class_name]

def to_yolo_format(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    return x_center, y_center, width, height


def filter_fully_overlapping_bboxes(bboxes, iou_threshold=0.3):
    """완전히(or 거의) 겹치는 박스 중 가장 큰 것만 남김"""
    def iou(a, b):
        xa1, ya1 = a['x'], a['y']
        xa2, ya2 = a['x'] + a['w'], a['y'] + a['h']
        xb1, yb1 = b['x'], b['y']
        xb2, yb2 = b['x'] + b['w'], b['y'] + b['h']

        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = a['w'] * a['h']
        area_b = b['w'] * b['h']
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    kept = []
    used = [False] * len(bboxes)

    for i in range(len(bboxes)):
        if used[i]:
            continue
        group = [i]
        for j in range(i + 1, len(bboxes)):
            if used[j]:
                continue
            if iou(bboxes[i], bboxes[j]) >= iou_threshold:
                group.append(j)

        if len(group) == 1:
            kept.append(bboxes[group[0]])
        else:
            # 여기만 수정: smallest_idx → largest_idx
            largest_idx = max(group, key=lambda k: bboxes[k]['w'] * bboxes[k]['h'])
            kept.append(bboxes[largest_idx])

        for k in group:
            used[k] = True

    return kept

def process_files(file_list, image_src_dir, image_dst_dir, label_dst_dir):
    for json_file in file_list:
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        image = data['description']['image']

        # 📌 원본 이미지 크기를 JSON이 아니라 실제 파일에서 읽음
        src_image_path = os.path.join(image_src_dir, image)
        img = cv2.imread(src_image_path)
        if img is None:
            print(f"Warning: Image not found, skipping: {src_image_path}")
            continue

        img_height, img_width = img.shape[:2]   # ← JSON width/height 무시하고 실제 크기 사용!

        # 메타데이터에서 클래스명 추출
        extracted = {item['name']: item['value'] for item in data.get('metadata', [])}
        part_name = extracted.get('작물부위코드', '').lstrip('\ufeff')
        class_name = extracted.get('작물상태코드', '').lstrip('\ufeff')

        if part_name != '열매':
            continue
        if not class_name:
            continue

        class_id = get_or_create_class_id(class_name)

        # bbox 읽기
        bbox_list = []
        for item in data.get('result', []):
            if item.get('type') == 'bbox':
                bbox_list.append({
                    'x': item.get('x'),
                    'y': item.get('y'),
                    'w': item.get('w'),
                    'h': item.get('h')
                })

        # 중복 bbox 제거
        bbox_list = filter_fully_overlapping_bboxes(bbox_list)

        # 이미지 복사
        dst_image_path = os.path.join(image_dst_dir, image)
        shutil.copy2(src_image_path, dst_image_path)

        # YOLO 라벨 저장
        image_basename = os.path.splitext(image)[0]
        label_path = os.path.join(label_dst_dir, image_basename + '.txt')

        with open(label_path, 'w', encoding='utf-8') as f:
            for bbox in bbox_list:
                x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                x_c, y_c, w_norm, h_norm = to_yolo_format(x, y, w, h, img_width, img_height)
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        print(f"Processed {image} ({part_name}, {class_name}) and saved annotation.")



# 데이터 처리
process_files(training_files, train_image_src_dir, train_image_dst_dir, train_label_dst_dir)
process_files(validation_files, val_image_src_dir, val_image_dst_dir, val_label_dst_dir)


# data.yaml 생성
data_yaml_path = 'Yolo/data.yaml'

id_to_class_name = [None] * len(class_name_to_id)
for name, idx in class_name_to_id.items():
    id_to_class_name[idx] = name

data_yaml = {
    'train': 'train/images',
    'val': 'val/images',
    'nc': len(id_to_class_name),
    'names': id_to_class_name
}

with open(data_yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(data_yaml, f, allow_unicode=True)

print(f"data.yaml 생성 완료: {data_yaml_path}")
print("클래스 매핑:", class_name_to_id)

