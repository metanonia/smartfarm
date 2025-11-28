import os
import shutil
import glob
import json
import yaml
import cv2

# ✅ 파일 리스트 가져오기 - 기존 + 신규 데이터
training_file_pattern_old = 'Data/Json/Training/TL_딸기_병해충피해이미지/*.json'
validation_file_pattern_old = 'Data/Json/Validation/VL_딸기_병해충피해이미지/*.json'

training_file_pattern_strawberry = 'Data/Json/Training/[라벨]04.딸기_1.질병/*.json'
validation_file_pattern_strawberry = 'Data/Json/Validation/[라벨]04.딸기_1.질병/*.json'


# 신규 설향 병해 4종 (역병, 시들음병, 잎끝마름, 황화)
training_file_patterns_new = [
    'Data/Json/Training/TL_01.딸기_001.설향_02.역병/*.json',
    'Data/Json/Training/TL_01.딸기_001.설향_03.시들음병/*.json',
    'Data/Json/Training/TL_01.딸기_001.설향_04.잎끝마름/*.json',
    'Data/Json/Training/TL_01.딸기_001.설향_05.황화/*.json',
]
validation_file_patterns_new = [
    'Data/Json/Validation/VL_01.딸기_001.설향_02.역병/*.json',
    'Data/Json/Validation/VL_01.딸기_001.설향_03.시들음병/*.json',
    'Data/Json/Validation/VL_01.딸기_001.설향_04.잎끝마름/*.json',
    'Data/Json/Validation/VL_01.딸기_001.설향_05.황화/*.json',
]

# training_files = glob.glob(training_file_pattern_old, recursive=True)
# training_files += glob.glob(training_file_pattern_strawberry, recursive=True)
# for p in training_file_patterns_new:
#     training_files += glob.glob(p, recursive=True)
#
# validation_files = glob.glob(validation_file_pattern_old, recursive=True)
# validation_files += glob.glob(validation_file_pattern_strawberry, recursive=True)
# for p in validation_file_patterns_new:
#     validation_files += glob.glob(p, recursive=True)

# 경로 설정
train_image_src_dirs = {
    'old': 'Data/Images/Training/TS_딸기_병해충피해이미지/',
    'strawberry': 'Data/Images/Training/04.딸기_1.질병/',
    'new_02': 'Data/Images/Training/TS_01.딸기_001.설향_02.역병/',
    'new_03': 'Data/Images/Training/TS_01.딸기_001.설향_03.시들음병/',
    'new_04': 'Data/Images/Training/TS_01.딸기_001.설향_04.잎끝마름/',
    'new_05': 'Data/Images/Training/TS_01.딸기_001.설향_05.황화/',
}
val_image_src_dirs = {
    'old': 'Data/Images/Validation/VS_딸기_병해충피해이미지/',
    'strawberry': 'Data/Images/Validation/04.딸기_1.질병/',
    'new_02': 'Data/Images/Validation/VS_01.딸기_001.설향_02.역병/',
    'new_03': 'Data/Images/Validation/VS_01.딸기_001.설향_03.시들음병/',
    'new_04': 'Data/Images/Validation/VS_01.딸기_001.설향_04.잎끝마름/',
    'new_05': 'Data/Images/Validation/VS_01.딸기_001.설향_05.황화/',
}

train_image_dst_dir = 'Yolo/train/images'
train_label_dst_dir = 'Yolo/train/labels'

val_image_dst_dir = 'Yolo/val/images'
val_label_dst_dir = 'Yolo/val/labels'

os.makedirs(train_image_dst_dir, exist_ok=True)
os.makedirs(train_label_dst_dir, exist_ok=True)
os.makedirs(val_image_dst_dir, exist_ok=True)
os.makedirs(val_label_dst_dir, exist_ok=True)

# ✅ 클래스명과 인덱스 매핑 - 신규 클래스 추가
class_mapping = {
    '열매_잿빛곰팡이병': 0,
    '열매_흰가루병': 1,
    '잎_흰가루병': 2,
    '잎_역병': 3,
    '잎_시들음병': 4,
    '잎_잎끝마름': 5,
    '잎_황화': 6,
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

# 병해별 너비/높이 필터 값 매핑
# 훈련시에는 큰 바운딩 박스만 고려
training_thresholds = {
    '잎_역병': 200,
    '잎_시들음병': 250,
    '잎_잎끝마름': 300,
    '잎_황화': 300,
}
# 검증시에는 작은 바운딩 박스도 함께 확인
# 훈련과 동일한 값을 지정해서 R값 확인
# 0 또는 작은값을 지정해서 P값 확인
validation_thresholds = {
    '잎_역병': 200,
    '잎_시들음병': 250,
    '잎_잎끝마름': 300,
    '잎_황화': 300,

}


def process_strawberry_files(file_list, image_src_dir, image_dst_dir, label_dst_dir):
    """04.딸기_1.질병 형식 처리"""
    disease_mapping = {7: '열매_잿빛곰팡이병', 8: '열매_흰가루병'}

    for json_file in file_list:
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        image_filename = data['description']['image']
        src_image_path = os.path.join(image_src_dir, image_filename)
        img = cv2.imread(src_image_path)

        if img is None:
            print(f"Warning: Image not found: {src_image_path}")
            continue

        img_height, img_width = img.shape[:2]
        disease_code = data['annotations']['disease']
        class_name = disease_mapping.get(disease_code, None)

        if class_name is None:
            print(f"Warning: Unknown disease code {disease_code} in {image_filename}")
            continue

        class_id = class_mapping[class_name]

        # points에서 bbox 추출 (xtl, ytl, xbr, ybr → x,y,w,h)
        bbox_list = []
        for point in data['annotations']['points']:
            xtl, ytl = point['xtl'], point['ytl']
            xbr, ybr = point['xbr'], point['ybr']
            w, h = xbr - xtl, ybr - ytl
            bbox_list.append({'x': xtl, 'y': ytl, 'w': w, 'h': h})

        bbox_list = filter_fully_overlapping_bboxes(bbox_list, iou_threshold=0.3)

        dst_image_path = os.path.join(image_dst_dir, image_filename)
        shutil.copy2(src_image_path, dst_image_path)

        image_basename = os.path.splitext(image_filename)[0]
        label_path = os.path.join(label_dst_dir, image_basename + '.txt')

        with open(label_path, 'w', encoding='utf-8') as f:
            for bbox in bbox_list:
                x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                x_c, y_c, w_norm, h_norm = to_yolo_format(x, y, w, h, img_width, img_height)
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        print(f"Processed strawberry {image_filename} as {class_name}")

def process_new_format_files(file_list, image_src_dir, image_dst_dir, label_dst_dir, leaf_class_key, size_thresholds):
    class_id = class_mapping[leaf_class_key]
    threshold = size_thresholds.get(leaf_class_key, 0)  # 없으면 0 (필터 없음)

    for json_file in file_list:
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        image = data['images']['fname']
        src_image_path = os.path.join(image_src_dir, image)
        img = cv2.imread(src_image_path)

        if img is None:
            print(f"Warning: Image file not found, skipping: {src_image_path}")
            continue

        img_height, img_width = img.shape[:2]

        # annotations에서 bbox 추출 - 잎만 필터링
        bbox_list = []
        for annotation in data.get('annotations', []):
            bbox = annotation['bbox']  # [x, y, width, height]
            category_name = next(
                (cat['name'] for cat in data['categories'] if cat['id'] == annotation['category_id']),
                None
            )

            if category_name == '잎':
                w = bbox[2]
                h = bbox[3]
                # 너비와 높이 자동 필터링 적용
                if w >= threshold and h >= threshold:
                    bbox_dict = {
                        'x': bbox[0],
                        'y': bbox[1],
                        'w': w,
                        'h': h,
                    }
                    bbox_list.append(bbox_dict)

        if len(bbox_list) == 0:
            print(f"Warning: No '잎' bbox meeting size threshold in {image}, skipping")
            continue

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

        print(f"Processed (new format) {image} as {leaf_class_key} with size filter {threshold} and saved annotation.")


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

print("Processing 04.딸기_1.질병 training data...")
strawberry_training_files = glob.glob(training_file_pattern_strawberry, recursive=True)
process_strawberry_files(strawberry_training_files, train_image_src_dirs['strawberry'],
                         train_image_dst_dir, train_label_dst_dir)

print("Processing new format training data...")

# 역병
new_training_files_02 = glob.glob(training_file_patterns_new[0], recursive=True)
process_new_format_files(new_training_files_02, train_image_src_dirs['new_02'],
                         train_image_dst_dir, train_label_dst_dir, '잎_역병', training_thresholds)

# 시들음병
new_training_files_03 = glob.glob(training_file_patterns_new[1], recursive=True)
process_new_format_files(new_training_files_03, train_image_src_dirs['new_03'],
                         train_image_dst_dir, train_label_dst_dir, '잎_시들음병', training_thresholds)

# 잎끝마름
new_training_files_04 = glob.glob(training_file_patterns_new[2], recursive=True)
process_new_format_files(new_training_files_04, train_image_src_dirs['new_04'],
                         train_image_dst_dir, train_label_dst_dir, '잎_잎끝마름', training_thresholds)

# 황화
new_training_files_05 = glob.glob(training_file_patterns_new[3], recursive=True)
process_new_format_files(new_training_files_05, train_image_src_dirs['new_05'],
                         train_image_dst_dir, train_label_dst_dir, '잎_황화', training_thresholds)

print("Processing validation data...")
old_validation_files = glob.glob(validation_file_pattern_old, recursive=True)
process_old_format_files(old_validation_files, val_image_src_dirs['old'],
                         val_image_dst_dir, val_label_dst_dir)

# ✅ 검증 데이터 처리 - 딸기 데이터 추가
print("Processing 04.딸기_1.질병 validation data...")
strawberry_validation_files = glob.glob(validation_file_pattern_strawberry, recursive=True)
process_strawberry_files(strawberry_validation_files, val_image_src_dirs['strawberry'],
                         val_image_dst_dir, val_label_dst_dir)

# 역병
new_validation_files_02 = glob.glob(validation_file_patterns_new[0], recursive=True)
process_new_format_files(new_validation_files_02, val_image_src_dirs['new_02'],
                         val_image_dst_dir, val_label_dst_dir, '잎_역병', validation_thresholds)

# 시들음병
new_validation_files_03 = glob.glob(validation_file_patterns_new[1], recursive=True)
process_new_format_files(new_validation_files_03, val_image_src_dirs['new_03'],
                         val_image_dst_dir, val_label_dst_dir, '잎_시들음병', validation_thresholds)

# 잎끝마름
new_validation_files_04 = glob.glob(validation_file_patterns_new[2], recursive=True)
process_new_format_files(new_validation_files_04, val_image_src_dirs['new_04'],
                         val_image_dst_dir, val_label_dst_dir, '잎_잎끝마름', validation_thresholds)

# 황화
new_validation_files_05 = glob.glob(validation_file_patterns_new[3], recursive=True)
process_new_format_files(new_validation_files_05, val_image_src_dirs['new_05'],
                         val_image_dst_dir, val_label_dst_dir, '잎_황화', validation_thresholds)

# data.yaml 생성
data_yaml_path = 'Yolo/data.yaml'
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

