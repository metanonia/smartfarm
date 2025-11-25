import os
import shutil
import glob
import json
import yaml
import cv2

# ✅ 파일 리스트 가져오기 - 기존 + 신규 데이터
training_file_pattern_old = 'Data/Json/Training/TL_딸기_병해충피해이미지/*.json'
validation_file_pattern_old = 'Data/Json/Validation/VL_딸기_병해충피해이미지/*.json'
training_file_pattern_new = 'Data/Json/Training/TL_01.딸기_001.설향_02.역병/*.json'
validation_file_pattern_new = 'Data/Json/Validation/VL_01.딸기_001.설향_02.역병/*.json'

training_files = glob.glob(training_file_pattern_old, recursive=True) + \
                 glob.glob(training_file_pattern_new, recursive=True)
validation_files = glob.glob(validation_file_pattern_old, recursive=True) + \
                 glob.glob(validation_file_pattern_new, recursive=True)

# 경로 설정
train_image_src_dirs = {
    'old': 'Data/Images/Training/TS_딸기_병해충피해이미지/',
    'new': 'Data/Images/Training/TS_01.딸기_001.설향_02.역병/'
}
val_image_src_dirs = {
    'old': 'Data/Images/Validation/VS_딸기_병해충피해이미지/',
    'new': 'Data/Images/Training/VS_01.딸기_001.설향_02.역병/'
}
train_image_dst_dir = 'Yolo4/train/images'
train_label_dst_dir = 'Yolo4/train/labels'

val_image_dst_dir = 'Yolo4/val/images'
val_label_dst_dir = 'Yolo4/val/labels'

os.makedirs(train_image_dst_dir, exist_ok=True)
os.makedirs(train_label_dst_dir, exist_ok=True)
os.makedirs(val_image_dst_dir, exist_ok=True)
os.makedirs(val_label_dst_dir, exist_ok=True)

# ✅ 클래스명과 인덱스 매핑 - 신규 클래스 추가
class_mapping = {
    '열매_잿빛곰팡이병': 0,
    '열매_흰가루병': 1,
    '잎_흰가루병': 2,
    '잎_역병': 3,  # 신규 추가
}


def to_yolo_format(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    return x_center, y_center, width, height


def filter_fully_overlapping_bboxes(bboxes, iou_threshold=0.3):
    """겹치는 박스 중 가장 큰 것만 남김"""

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

    if len(bboxes) == 0:
        return []

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
            largest_idx = max(group, key=lambda k: bboxes[k]['w'] * bboxes[k]['h'])
            kept.append(bboxes[largest_idx])

        for k in group:
            used[k] = True

    return kept


# ✅ 신규 데이터 처리 함수 (annotations 형식)
def process_new_format_files(file_list, image_src_dir, image_dst_dir, label_dst_dir):
    for json_file in file_list:
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        # 이미지 파일명 추출
        image = data['images']['fname']

        src_image_path = os.path.join(image_src_dir, image)
        img = cv2.imread(src_image_path)

        if img is None:
            print(f"Warning: Image file not found, skipping: {src_image_path}")
            continue

        img_height, img_width = img.shape[:2]

        # ✅ annotations에서 bbox 추출 - 잎만 필터링
        bbox_list = []
        for annotation in data.get('annotations', []):
            bbox = annotation['bbox']  # [x, y, width, height]
            category_name = next(
                (cat['name'] for cat in data['categories'] if cat['id'] == annotation['category_id']),
                None
            )

            # ✅ 잎만 선택
            if category_name == '잎':
                bbox_dict = {
                    'x': bbox[0],
                    'y': bbox[1],
                    'w': bbox[2],
                    'h': bbox[3],
                }
                bbox_list.append(bbox_dict)

        if len(bbox_list) == 0:
            print(f"Warning: No '잎' bbox found in {image}, skipping")
            continue

        # ✅ 중복 필터링 - 가장 큰 bbox만 남김
        bbox_list = filter_fully_overlapping_bboxes(bbox_list, iou_threshold=0.3)

        # 이미지 복사
        dst_image_path = os.path.join(image_dst_dir, image)
        shutil.copy2(src_image_path, dst_image_path)

        # annotation 파일 생성
        image_basename = os.path.splitext(image)[0]
        label_path = os.path.join(label_dst_dir, image_basename + '.txt')

        class_id = class_mapping['잎_역병']  # 신규 클래스

        with open(label_path, 'w', encoding='utf-8') as f:
            for bbox in bbox_list:
                x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                x_c, y_c, w_norm, h_norm = to_yolo_format(x, y, w, h, img_width, img_height)
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        print(f"Processed (new format) {image} and saved annotation.")


# ✅ 기존 데이터 처리 함수 (metadata 형식)
def process_old_format_files(file_list, image_src_dir, image_dst_dir, label_dst_dir):
    for json_file in file_list:
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        image = data['description']['image']
        src_image_path = os.path.join(image_src_dir, image)
        img = cv2.imread(src_image_path)

        if img is None:
            print(f"Warning: Image file not found, skipping: {src_image_path}")
            continue

        img_height, img_width = img.shape[:2]

        extracted = {item['name']: item['value'] for item in data.get('metadata', [])}
        part_name = extracted.get('작물부위코드', '').lstrip('\ufeff')
        class_name = extracted.get('작물상태코드', '').lstrip('\ufeff')

        full_class_name = f"{part_name}_{class_name}"
        class_id = class_mapping.get(full_class_name, -1)

        if class_id == -1:
            print(f"Warning: Unknown class name: {full_class_name}, skipping")
            continue

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

        bbox_list = filter_fully_overlapping_bboxes(bbox_list, iou_threshold=0.3)

        dst_image_path = os.path.join(image_dst_dir, image)
        shutil.copy2(src_image_path, dst_image_path)

        image_basename = os.path.splitext(image)[0]
        label_path = os.path.join(label_dst_dir, image_basename + '.txt')

        with open(label_path, 'w', encoding='utf-8') as f:
            for bbox in bbox_list:
                x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                x_c, y_c, w_norm, h_norm = to_yolo_format(x, y, w, h, img_width, img_height)
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        print(f"Processed (old format) {image} and saved annotation.")


# ✅ 학습 데이터 처리 - 기존 + 신규 분리 처리
print("Processing old format training data...")
old_training_files = glob.glob(training_file_pattern_old, recursive=True)
process_old_format_files(old_training_files, train_image_src_dirs['old'],
                         train_image_dst_dir, train_label_dst_dir)

print("Processing new format training data...")
new_training_files = glob.glob(training_file_pattern_new, recursive=True)
process_new_format_files(new_training_files, train_image_src_dirs['new'],
                         train_image_dst_dir, train_label_dst_dir)

print("Processing validation data...")
old_validatioin_files = glob.glob(validation_file_pattern_old, recursive=True)
process_old_format_files(old_validatioin_files, val_image_src_dirs['old'],
                         val_image_dst_dir, val_label_dst_dir)
new_validation_files = glob.glob(validation_file_pattern_new, recursive=True)
process_new_format_files(new_validation_files, val_image_src_dirs['new'],
                         train_image_dst_dir, train_label_dst_dir)

# data.yaml 생성
data_yaml_path = 'Yolo4/data.yaml'
data_yaml = {
    'train': 'train/images',
    'val': 'val/images',
    'nc': len(class_mapping),
    'names': [k for k, v in sorted(class_mapping.items(), key=lambda item: item[1])]
}

with open(data_yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(data_yaml, f, allow_unicode=True)

print(f"data.yaml 생성 완료: {data_yaml_path}")
print(f"Total classes: {len(class_mapping)}")
print(f"Class mapping: {class_mapping}")
