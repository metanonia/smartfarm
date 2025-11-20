import os
import shutil
import glob
import json
from PIL import Image

# 파일 리스트 가져오기
training_file_pattern = 'Data/Json/Training/TL_딸기_병해충피해이미지/*.json'
validation_file_pattern = 'Data/Json/Validation/VL_딸기_병해충피해이미지/*.json'
training_files = glob.glob(training_file_pattern, recursive=True)
validation_files = glob.glob(validation_file_pattern, recursive=True)

# 경로 설정
train_image_src_dir = 'Data/Images/Training/TS_딸기_병해충피해이미지/'
val_image_src_dir = 'Data/Images/Validation/VS_딸기_병해충피해이미지/'

# 분류용 출력 폴더 (클래스별 하위 폴더)
train_cls_dst_dir = 'Yolo_Classification/train'
val_cls_dst_dir = 'Yolo_Classification/val'

# 클래스명과 인덱스 매핑
class_mapping = {
    '열매_잿빛곰팡이병': 0,
    '열매_흰가루병': 1,
    '잎_흰가루병': 2,
}

# 클래스별 폴더 생성
for split_dir in [train_cls_dst_dir, val_cls_dst_dir]:
    for cls_name in class_mapping.keys():
        os.makedirs(os.path.join(split_dir, cls_name), exist_ok=True)


def process_classification_files(file_list, image_src_dir, cls_dst_dir):
    crop_count = 0
    for json_file in file_list:
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        image_name = data['description']['image']
        img_width = data['description']['width']
        img_height = data['description']['height']

        extracted = {item['name']: item['value'] for item in data.get('metadata', [])}
        part_name = extracted.get('작물부위코드', '').lstrip('\ufeff')
        class_name = extracted.get('작물상태코드', '').lstrip('\ufeff')

        full_class_name = f"{part_name}_{class_name}"
        if full_class_name not in class_mapping:
            print(f"Warning: Unknown class {full_class_name}, skipping")
            continue

        bbox_list = []
        for item in data.get('result', []):
            if item.get('type') == 'bbox':
                bbox_list.append({
                    'x': item.get('x'),
                    'y': item.get('y'),
                    'w': item.get('w'),
                    'h': item.get('h')
                })

        src_image_path = os.path.join(image_src_dir, image_name)
        if not os.path.isfile(src_image_path):
            print(f"Warning: Image file not found, skipping: {src_image_path}")
            continue

        # 이미지 열기
        img = Image.open(src_image_path)

        # bbox별로 크롭 및 저장
        for idx, bbox in enumerate(bbox_list):
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

            # 크롭 영역 (left, upper, right, lower)
            crop_box = (x, y, x + w, y + h)
            cropped_img = img.crop(crop_box)

            # 저장 파일명: 원본파일명_bbox인덱스.jpg
            image_basename = os.path.splitext(image_name)[0]
            crop_filename = f"{image_basename}_crop{idx}.jpg"
            crop_save_path = os.path.join(cls_dst_dir, full_class_name, crop_filename)

            cropped_img.save(crop_save_path)
            crop_count += 1

        print(f"Processed {image_name}, cropped {len(bbox_list)} regions.")

    print(f"Total cropped images: {crop_count}")


# 학습/검증 데이터 처리
print("Processing training data...")
process_classification_files(training_files, train_image_src_dir, train_cls_dst_dir)

print("Processing validation data...")
process_classification_files(validation_files, val_image_src_dir, val_cls_dst_dir)

print("Classification dataset creation completed!")
