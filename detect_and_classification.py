import os
import json
import cv2
from PIL import Image
from ultralytics import YOLO

# 모델 로드
detection_model = YOLO('Models/detection_model.pt')  # 병 유무 탐지 모델
classification_model = YOLO('Models/classification_model.pt')  # 병명 분류 모델

# 클래스 매핑 (분류 모델용)
class_names = ['열매_잿빛곰팡이병', '열매_흰가루병', '잎_흰가루병']

# 테스트 이미지 및 JSON 경로
test_image_dir = 'Data/Images/Validation/VS_딸기_병해충피해이미지'
test_json_dir = 'Data/Json/Validation/VL_딸기_병해충피해이미지'


def load_ground_truth(json_path):
    """JSON 파일에서 정답 로드"""
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)

    extracted = {item['name']: item['value'] for item in data.get('metadata', [])}
    part_name = extracted.get('작물부위코드', '').lstrip('\ufeff')
    class_name = extracted.get('작물상태코드', '').lstrip('\ufeff')

    ground_truth = f"{part_name}_{class_name}"

    # bbox 정보도 가져오기
    bbox_list = []
    for item in data.get('result', []):
        if item.get('type') == 'bbox':
            bbox_list.append({
                'x': item.get('x'),
                'y': item.get('y'),
                'w': item.get('w'),
                'h': item.get('h')
            })

    return ground_truth, bbox_list


def detect_and_classify(image_path, json_path):
    """탐지 + 분류 + 정답 비교"""
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image {image_path}")
        return

    # JSON에서 정답 로드
    ground_truth, gt_bboxes = load_ground_truth(json_path)
    print(f"\n{'=' * 60}")
    print(f"Processing: {os.path.basename(image_path)}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Ground Truth BBoxes: {len(gt_bboxes)} boxes")

    # 1단계: 탐지 (병 유무 검출)
    detection_results = detection_model.predict(image_path, conf=0.3)

    if len(detection_results[0].boxes) == 0:
        print("No disease detected.")
        return

    print(f"Detected {len(detection_results[0].boxes)} regions")

    # 2단계: 각 검출 영역을 크롭하여 분류
    predictions = []
    for i, box in enumerate(detection_results[0].boxes):
        # bbox 좌표 (xyxy 형식)
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        # 크롭
        cropped = img[y1:y2, x1:x2]

        # PIL 형식으로 변환
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        # 분류 모델 실행
        cls_results = classification_model.predict(cropped_pil, verbose=False)

        # 예측 클래스
        pred_cls_id = int(cls_results[0].probs.top1)
        pred_cls_name = class_names[pred_cls_id]
        pred_conf = float(cls_results[0].probs.top1conf)

        predictions.append({
            'bbox': (x1, y1, x2, y2),
            'class': pred_cls_name,
            'confidence': pred_conf
        })

        print(f"  Region {i + 1}: {pred_cls_name} (conf: {pred_conf:.2f})")

        # 시각화용 박스 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{pred_cls_name} {pred_conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # 정답과 비교
    correct = 0
    for pred in predictions:
        if pred['class'] == ground_truth:
            correct += 1

    accuracy = correct / len(predictions) if predictions else 0
    print(f"\nAccuracy: {correct}/{len(predictions)} ({accuracy * 100:.1f}%)")

    # 결과 이미지 저장
    output_dir = 'Results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Result saved to: {output_path}")

    return predictions, ground_truth, accuracy


def process_test_dataset(image_dir, json_dir):
    """전체 테스트 데이터셋 처리"""
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    total_correct = 0
    total_predictions = 0

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        json_file = os.path.splitext(image_file)[0] + '.json'
        json_path = os.path.join(json_dir, json_file)

        if not os.path.exists(json_path):
            print(f"Warning: JSON not found for {image_file}")
            continue

        predictions, ground_truth, accuracy = detect_and_classify(image_path, json_path)

        if predictions:
            correct = sum(1 for p in predictions if p['class'] == ground_truth)
            total_correct += correct
            total_predictions += len(predictions)

    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"Overall Accuracy: {total_correct}/{total_predictions} ({overall_accuracy * 100:.1f}%)")


# 단일 이미지 테스트
if __name__ == "__main__":
    # 단일 이미지 테스트
    test_image = 'Data/Images/Validation/VS_딸기_병해충피해이미지/V003_5_2_1_2_4_3_4_1_0_0_20221210_4458_20240422174546.jpg'
    test_json = 'Data/Json/Validation/VL_딸기_병해충피해이미지/V003_5_2_1_2_4_3_4_1_0_0_20221210_4458_20240422174546.json'

    if os.path.exists(test_image) and os.path.exists(test_json):
        detect_and_classify(test_image, test_json)

    # 전체 데이터셋 테스트 (옵션)
    # process_test_dataset(test_image_dir, test_json_dir)
