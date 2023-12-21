import cv2
import os

import numpy as np
from PIL import Image
from ultralytics import YOLO
import timm
import torch
import torchvision.transforms as transforms

from diseases import edward, vibrio, strepto, tenaci, entero, \
    miamien, vhsv, diseases, init_diseases
from symptom import GillSymptom, LiverSymptom, GillCoverSymptom, \
    IntestineSymptom, AscitesSymptom, FRSymptom, RSSymptom

fish_img_path = '/media/lifeisbug/hdd/fish/fish_data/flexink_8+9/images/'
model_path = 'final_predict/weights/'

def segmentation(model, image):
    # GPU 사용 가능 여부 확인
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)

    segmentation_results = model.predict(image)

    masks = segmentation_results[0].masks

    if masks is None:
        return None

    # 주어진 마스크 텐서 값
    mask_tensor = masks.data

    # 마스크 텐서 값을 넘파이 배열로 변환
    mask_numpy_array = mask_tensor.cpu().numpy()

    # 원본 이미지와 동일한 크기로 마스크 조정
    resized_mask = np.array(Image.fromarray(
        mask_numpy_array[0]).resize((image.shape[1], image.shape[0])))

    # 이진 마스크 생성
    binary_mask = (resized_mask > 0).astype(np.uint8)

    # 이진 마스크를 이미지 크기에 맞게 확장
    expanded_mask = np.expand_dims(binary_mask, axis=-1)

    # 마스크에 해당하는 이미지 추출
    masked_image = image * expanded_mask

    # 이미지 저장
    masked_image = Image.fromarray(masked_image.astype(np.uint8))
    masked_image_resized = masked_image.resize((224, 224))

    return masked_image_resized


def detection(model, image):
    # GPU 사용 가능 여부 확인
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)

    preds = model(image)
    cls_cnt = [0, 0, 0, 0, 0, 0, 0]  # 각 클래스가 검출된 개수
    for pred in preds:
        boxes = pred.boxes
        for box in boxes:
            cls = int(box.cls)
            cls_cnt[cls] += 1

    return cls_cnt


def classification(model, masked_image):
    # 이미지 전처리를 위한 변환 정의
    segmentation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # GPU 사용 가능 여부 확인
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 입력 이미지 로드
    masked_image_tensor = segmentation_transform(
        masked_image).unsqueeze(0).to(device)

    # 분류 모델을 사용하여 분류 수행
    with torch.no_grad():
        outputs = model(masked_image_tensor)
        _, predicted = torch.max(outputs.data, 1)

    # 분류 결과
    return predicted.item()


def set_symptom_in_disease(symptom_type, *args):
    target_disease = []
    for arg in args:
        target_disease.append(arg)

    for disease in diseases:
        for symptoms in disease.keys():
            if not target_disease and symptom_type in symptoms:
                disease[symptoms] = True

            else:
                for target in target_disease:
                    if target in symptoms and symptom_type in symptoms:
                        disease[symptoms] = True


def compare(pred_gill,
            pred_liver,
            pred_gill_cover,
            pred_intestine,
            pred_ascites,
            pred_eye_90,
            pred_eyeless_90):

    symptom = []

    if pred_gill == GillSymptom.PANMY.value:
        print("아가미 이미지는 빈혈 아가미 클래스로 예측됩니다.")
        symptom.append('아가미 빈혈')
        set_symptom_in_disease('아가미')
    elif pred_gill == GillSymptom.NORMAL.value:
        print('아가미 이미지는 정상 아가미 클래스로 예측됩니다.')

    if pred_liver == LiverSymptom.CONGEST.value:
        print("간 이미지는 울혈 클래스로 예측됩니다.")
        symptom.append('간 울혈')
        set_symptom_in_disease('간', '비브리오', '연쇄구균', '바이러스성')
    elif pred_liver == LiverSymptom.NORMAL.value:
        print("간 이미지는 정상 클래스로 예측됩니다.")
    elif pred_liver == LiverSymptom.INFLAM.value:
        print("간 이미지는 염증 클래스로 예측됩니다.")
        symptom.append('간 염증')
        set_symptom_in_disease('간', '비브리오')
    elif pred_liver == LiverSymptom.PANMY.value:
        print("간 이미지는 빈혈 클래스로 예측됩니다.")
        symptom.append('간 빈혈')
        set_symptom_in_disease('간', '에드워드', '연쇄구균')

    if pred_gill_cover == GillCoverSymptom.NORMAL.value:
        print("아가미 뚜껑 이미지는 정상 클래스로 예측됩니다.")
    elif pred_gill_cover == GillCoverSymptom.HEMORR.value:
        print("아가미 뚜껑 이미지는 염증 및 출혈 클래스로 예측됩니다.")
        symptom.append('아가미뚜껑 염증 및 출혈')
        set_symptom_in_disease('아가미뚜껑', '연쇄구균')

    if pred_intestine == IntestineSymptom.COMPOUND.value:
        print("장 이미지는 염증 및 출혈 클래스로 예측됩니다.")
        symptom.append('장 염증 및 출혈')
        set_symptom_in_disease('장', '에드워드', '연쇄구균')
    elif pred_intestine == IntestineSymptom.NORMAL.value:
        print("장 이미지는 정상 클래스로 예측됩니다.")

    if pred_ascites == AscitesSymptom.NORMAL.value:
        print("복수 이미지는 정상 클래스로 예측됩니다.")
    elif pred_ascites == AscitesSymptom.HEMORR.value:
        print("복수 이미지는 출혈성 복수 클래스로 예측됩니다.")
        symptom.append('출혈성 복수')
        set_symptom_in_disease('복수', '에드워드', '연쇄구균')
    elif pred_ascites == AscitesSymptom.CLEAN.value:
        print("복수 이미지는 맑은 복수 클래스로 예측됩니다.")
        symptom.append('맑은 복수')
        set_symptom_in_disease('복수', '바이러스성')

    if pred_eye_90[FRSymptom.DYH.value] > 0:
        detected = '체표 출혈'
        print(f"유안측 이미지에서 {detected}이 발견됩니다.")
        symptom.append(detected)
        set_symptom_in_disease('체표', '비브리오', '연쇄구균')

    if pred_eye_90[FRSymptom.DYU.value] > 0:
        detected = '체표 궤양'
        print(f"유안측 이미지에서 {detected}이 발견됩니다.")
        symptom.append(detected)
        set_symptom_in_disease('체표', '비브리오')

    if pred_eye_90[FRSymptom.DYA.value] > 0:
        detected = '체표 근육 출혈'
        print(f"유안측 이미지에서 {detected}이 발견됩니다.")
        symptom.append(detected)
        set_symptom_in_disease('체표', '에드워드')

    if pred_eye_90[FRSymptom.FDH.value] > 0:
        detected = '등지느러미 출혈'
        print(f"유안측 이미지에서 {detected}이 발견됩니다.")
        symptom.append(detected)
        set_symptom_in_disease('체표', '비브리오', '연쇄구균', '스쿠티카')

    if pred_eye_90[FRSymptom.FAH.value] > 0:
        detected = '뒷지느러미 출혈'
        print(f"유안측 이미지에서 {detected}이 발견됩니다.")
        symptom.append(detected)
        set_symptom_in_disease('체표', '비브리오', '연쇄구균', '스쿠티카')

    if pred_eye_90[FRSymptom.FCH.value] > 0:
        detected = '꼬리지느러미 출혈'
        print(f"유안측 이미지에서 {detected}이 발견됩니다.")
        symptom.append(detected)
        set_symptom_in_disease('체표', '비브리오', '연쇄구균', '스쿠티카')

    if pred_eye_90[FRSymptom.MOU.value] > 0:
        detected = '주둥이 궤양'
        print(f"유안측 이미지에서 {detected}이 발견됩니다.")
        symptom.append(detected)
        set_symptom_in_disease('체표', '비브리오', '활주세균', '스쿠티카')

    if pred_eyeless_90[RSSymptom.DYH.value] > 0:
        detected = '체표 출혈'
        print(f"무안측 이미지에서 {detected}이 발견됩니다.")
        symptom.append(detected)
        set_symptom_in_disease('체표', '비브리오', '연쇄구균')

    if pred_eyeless_90[RSSymptom.DYU.value] > 0:
        detected = '체표 궤양'
        print(f"무안측 이미지에서 {detected}이 발견됩니다.")
        symptom.append(detected)
        set_symptom_in_disease('체표', '비브리오')

    if pred_eyeless_90[RSSymptom.DYA.value] > 0:
        detected = '체표 근육 출혈'
        print(f"무안측 이미지에서 {detected}이 발견됩니다.")
        symptom.append(detected)
        set_symptom_in_disease('체표', '에드워드')

    if pred_eyeless_90[RSSymptom.FDH.value] > 0:
        detected = '등지느러미 출혈'
        print(f"무안측 이미지에서 {detected}이 발견됩니다.")
        symptom.append(detected)
        set_symptom_in_disease('체표', '비브리오', '연쇄구균', '스쿠티카')

    if pred_eyeless_90[RSSymptom.FAH.value] > 0:
        detected = '뒷지느러미 출혈'
        print(f"무안측 이미지에서 {detected}이 발견됩니다.")
        symptom.append(detected)
        set_symptom_in_disease('체표', '비브리오', '연쇄구균', '스쿠티카')

    if pred_eyeless_90[RSSymptom.FCH.value] > 0:
        detected = '꼬리지느러미 출혈'
        print(f"무안측 이미지에서 {detected}이 발견됩니다.")
        symptom.append(detected)
        set_symptom_in_disease('체표', '비브리오', '연쇄구균', '스쿠티카')

    if pred_eyeless_90[RSSymptom.MOU.value] > 0:
        detected = '주둥이 궤양'
        print(f"무안측 이미지에서 {detected}이 발견됩니다.")
        symptom.append(detected)
        set_symptom_in_disease('체표', '비브리오', '활주세균', '스쿠티카')

    print(edward)
    edward_count = sum(value for value in edward.values() if value)
    print("에드워드증상개수:", edward_count)
    edward_symptom_rate = round((edward_count / len(edward) * 100), 2)
    print(f"에드워드증상/전체증상: {edward_symptom_rate}%", end='\n')

    print(vibrio)
    vibrio_count = sum(value for value in vibrio.values() if value)
    print("비브리오증상개수:", vibrio_count)
    vibrio_symptom_rate = round((vibrio_count / len(vibrio) * 100), 2)
    print(f"비브리오증상/전체증상: {vibrio_symptom_rate}%", end='\n')

    print(strepto)
    strepto_count = sum(value for value in strepto.values() if value)
    print("연쇄구균증상개수:", strepto_count)
    strepto_symptom_rate = round((strepto_count / len(strepto) * 100), 2)
    print(f"연쇄구균증상/전체증상: {strepto_symptom_rate}%", end='\n')

    print(tenaci)
    tenaci_count = sum(value for value in tenaci.values() if value)
    print("활주세균증상개수:", tenaci_count)
    tenaci_symptom_rate = round((tenaci_count / len(tenaci) * 100), 2)
    print(f"활주세균증상/전체증상: {tenaci_symptom_rate}%", end='\n')

    print(entero)
    entero_count = sum(value for value in entero.values() if value)
    print("여윔증증상개수:", entero_count)
    entero_symptom_rate = round((entero_count / len(entero) * 100), 2)
    print(f"여윔증증상/전체증상: {entero_symptom_rate}%", end='\n')

    print(miamien)
    miamien_count = sum(value for value in miamien.values() if value)
    print("스쿠티카증상개수:", miamien_count)
    miamien_symptom_rate = round((miamien_count / len(miamien) * 100), 2)
    print(f"스쿠티카증상/전체증상: {miamien_symptom_rate}%", end='\n')

    print(vhsv)
    vhsv_count = sum(value for value in vhsv.values() if value)
    print("바이러스성출혈성패혈증증상개수:", vhsv_count)
    vhsv_symptom_rate = round((vhsv_count / len(vhsv) * 100), 2)
    print(f"바이러스성출혈성패혈증증상/전체증상: {vhsv_symptom_rate}%", end='\n')

    symptom_rate = {
        '에드워드': edward_symptom_rate,
        '비브리오': vibrio_symptom_rate,
        '연쇄구균': strepto_symptom_rate,
        '활주세균': tenaci_symptom_rate,
        '여윔증': entero_symptom_rate,
        '스쿠티카': miamien_symptom_rate,
        "바이러스성출혈성패혈증": vhsv_symptom_rate
    }

    print(f"발견된 증상 : {symptom}")

    # 딕셔너리를 값에 따라 내림차순으로 정렬
    sorted_sympton = sorted(symptom_rate.items(), key=lambda x: x[1], reverse=True)
    
    # 상위 3개 값을 출력
    if sorted_sympton[0][1] < 20:
        print("해당 넙치는 정상으로 예측됩니다.")
        return [('정상', 0)]
    else:
        top_3_symptoms = sorted_sympton[:3]
        for num, (symptom, value) in enumerate(top_3_symptoms, start=1):
            print(f"{num} : {symptom} , {value}%")
        return top_3_symptoms


def final_predict(fish):

    init_diseases()
    
    image_eye_0 = cv2.imread(fish + "_01.JPG", cv2.IMREAD_UNCHANGED)
    image_eyeless_0 = cv2.imread(fish + "_06.JPG", cv2.IMREAD_UNCHANGED)

    image_eye_90 = cv2.imread(fish + "_03.JPG", cv2.IMREAD_UNCHANGED)
    image_eyeless_90 = cv2.imread(fish + "_08.JPG", cv2.IMREAD_UNCHANGED)

    image_gill_liver = cv2.imread(fish + "_12.JPG", cv2.IMREAD_UNCHANGED)
    image_gill_cover = cv2.imread(fish + "_13.JPG", cv2.IMREAD_UNCHANGED)
    image_intest_ascites = cv2.imread(fish + "_15.JPG", cv2.IMREAD_UNCHANGED)

    # model_eye_det_0 = YOLO(model_path + 'det/eye_0_best.pt')
    # model_eyeless_det_0 = YOLO(model_path + 'det/eyeless_0_best.pt')
    model_eye_det_45_90 = YOLO(model_path + 'det/eye_45_90_best.pt')
    model_eyeless_det_45_90 = YOLO(model_path + 'det/eyeless_45_90_best.pt')

    model_gill_seg = YOLO(model_path + 'seg/gill_best.pt')
    model_liver_seg = YOLO(model_path + 'seg/liver_best.pt')
    model_gill_cover_seg = YOLO(model_path + 'seg/gill_cover_best.pt')
    model_intestine_seg = YOLO(model_path + 'seg/intestine_best.pt')
    model_ascites_seg = YOLO(model_path + 'seg/ascites_best.pt')

    model_gill_cls = timm.create_model('coatnet_2_rw_224', pretrained=False, num_classes=2)
    model_gill_cls.load_state_dict(torch.load(model_path + 'cls/2gill3-c2-f2.pth'))
    model_gill_cls.eval()

    model_liver_cls = timm.create_model('efficientnetv2_rw_s', pretrained=False, num_classes=4)
    model_liver_cls.load_state_dict(torch.load(model_path + 'cls/liver4-ev2s-f2.pth'))
    model_liver_cls.eval()

    model_gill_cover_cls = timm.create_model('efficientnetv2_rw_m', pretrained=False, num_classes=2)
    model_gill_cover_cls.load_state_dict(torch.load(model_path + 'cls/gillcap-ev2m-f1.pth'))
    model_gill_cover_cls.eval()

    model_ascites_cls = timm.create_model('dm_nfnet_f1', pretrained=False, num_classes=3)
    model_ascites_cls.load_state_dict(torch.load(model_path + 'cls/guts2-n1-f5.pth'))
    model_ascites_cls.eval()

    model_intestine_cls = timm.create_model('coatnet_0_rw_224', pretrained=False, num_classes=2)
    model_intestine_cls.load_state_dict(torch.load(model_path + 'cls/organ-c0-f2.pth'))
    model_intestine_cls.eval()

    org_gill = segmentation(model_gill_seg, image_gill_liver)
    org_liver = segmentation(model_liver_seg, image_gill_liver)
    org_gill_cover = segmentation(model_gill_cover_seg, image_gill_cover)
    org_intestine = segmentation(model_intestine_seg, image_intest_ascites)
    org_ascites = segmentation(model_ascites_seg, image_intest_ascites)

    pred_eye_90 = detection(model_eye_det_45_90, image_eye_90)
    pred_eyeless_90 = detection(model_eyeless_det_45_90, image_eyeless_90)

    if org_gill is not None:
        pred_gill = classification(model_gill_cls, org_gill)
    else:
        pred_gill = -1

    if org_liver is not None: 
        pred_liver = classification(model_liver_cls, org_liver)
    else:
        pred_liver = -1
    
    if org_gill_cover is not None:
        pred_gill_cover = classification(model_gill_cover_cls, org_gill_cover)
    else:
        pred_gill_cover = -1
    
    if org_intestine is not None:
        pred_intestine = classification(model_intestine_cls, org_intestine)
    else:
        pred_intestine = -1
    
    if org_ascites is not None:        
        pred_ascites = classification(model_ascites_cls, org_ascites)
    else:
        pred_ascites = -1

    return compare(pred_gill, pred_liver, pred_gill_cover, pred_intestine, pred_ascites, pred_eye_90, pred_eyeless_90)


if __name__ == '__main__':
    fish_num = input('개체번호 입력 : ')
    final_predict(fish_img_path + fish_num)
