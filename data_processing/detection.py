import os
import random
import shutil
import cv2
import albumentations as A

from utils import get_image, get_class, get_img_num
from images import show_img
from dicts import class_code, class_side_0, class_eye_45_90, class_eyeless_45_90
from segmentation import cvt_map
from directory import root_dir_08, root_dir_09
from tqdm import tqdm_notebook

# det_dir = "/media/lifeisbug/hdd/fish/fish_data/data_8+9/object_detection/"
def bbox_with_img(img, bbox, cls, dir, maps=None):
    """ 이미지에 바운딩 박스, 폴리곤을 그림
    
    Args:
        img: 이미지 파일 이름
        bbox: 바운딩 박스 좌표
        cls: 질병 이름
        dir: 데이터셋의 이미지 폴더 경로
        
    Returns:
    
    """
    origin = cv2.imread(dir + img, cv2.IMREAD_UNCHANGED)
    # show_img(origin, "original image")
    
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]) 
    visualized = cv2.rectangle(origin, (x, y), (x+w, y+h), red, 4)
    
    if maps:
        areas = cvt_map(maps)
        for area in areas: 
            visualized = cv2.polylines(visualized, [area], True, green, 2)
    
    # 텍스트 정보
    text = cls
    org = (x, y - 11)  # 텍스트 위치
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.2  # 텍스트 크기
    color = (255, 255, 255)  # 텍스트 색상 
    thickness = 3
    lineType = cv2.LINE_AA
    
    # 텍스트 크기 계산
    (text_width, text_height), _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
    
    # 텍스트 영역 계산
    top_left = (x - 3, y - 45)
    bottom_right = (org[0] + text_width, org[1] + text_height - 15)
    
    cv2.rectangle(visualized, top_left, bottom_right, red, -1)
    cv2.putText(visualized, text, org, fontFace, fontScale, color, thickness, lineType)
    
    show_img(visualized, img)
    print(cls, class_code[cls])
    

def visualize_box(data_img, data_ann, data_cls, data_dir):  
    """ 원본 이미지에 바운딩 박스, 폴리곤 시각화
        * 검출 이미지에 대해 두 라벨을 다 확인할 필요가 있어 추가함
    Args:
        data_img: 데이터셋의 이미지 정보
        data_ann: 데이터셋의 어노테이션 정보
        data_cls: 데이터셋의 카테고리 정보
        data_dir: 데이터셋의 이미지 폴더 경로
    
    Returns:
        
    """ 
    for ann in data_ann:
        
        img_name = get_image(data_img, ann['image_id'], 'file_name')
        
        if not ann['bbox']:
            continue
                    
        code = get_class(data_cls, ann['category_id'], 'code')
                
        # 객체 외형 바운딩 박스만 표시
        # if code.find('OUY') != -1 or code.find('OUN') != -1: 
        #     visualize_box(get_image(data_img, ann['image_id'], 'file_name'), 
        #                         ann['bbox'], 
        #                         code, 
        #                         data_dir)
        
        # 5월 데이터의 경우 맵이 'counts' 키와 연결되어 있음
        if ann['segmentation'] and 'counts' in ann['segmentation']:
            ann_maps = ann['segmentation']['counts']
        else:
            ann_maps = ann['segmentation']
        
        if ann['segmentation']:  
            bbox_with_img(img_name, ann['bbox'], code, data_dir, ann_maps)
        else:
            bbox_with_img(img_name, ann['bbox'], code, data_dir)
            

def coco_2_yolo(bbox, size):
    """ COCO 형식 bbox를 YOLO 형식의 bbox로 변환
    
    Args:
        bbox: COCO 형식 바운딩 박스 좌표 리스트(x_min, y_min, weight, height)
        size: 원본 이미지 사이즈 (width, height)
    
    Returns:
        YOLO 형식으로 변환된 바운딩 박스 좌표 리스트
        
    """
    dw = 1./size[0] 
    dh = 1./size[1] 

    w = bbox[2]
    h = bbox[3]
    x_center = (w / 2) + bbox[0]
    y_center = (h / 2) + bbox[1]
       
    x_center = round(x_center * dw, 6)
    y_center = round(y_center * dh, 6) 
    w = round(w * dw, 6) 
    h = round(h * dh, 6) 

    return [x_center,y_center,w,h] 


def get_bboxes():
    """
    
    Args:
        
    
    Returns:
        YOLO bbox의 리스트로 이루어진 2차원 리스트
    
    """

    
def save_yolo_bbox(img_name, bboxes, cls, size, aug_type="origin"):
    """ 바운딩 박스 좌표를 YOLO 형식으로 저장
    
    Args:
        img_name: 원본 이미지 파일 이름
        bboxes: yolo 바운딩 박스 좌표(2차원 리스트 형태)
        cls: 클래스 코드
        size: 원본 이미지 사이즈
        aug_type: 적용된 증강 종류(origin: 증강 미적용)
        
    Returns:
    
    """
    label_path = root_dir + 'object_detection' + '/labels/'
    
    cls_num = list(class_code.keys()).index(cls)
    content = [cls_num, ybox[0], ybox[1], ybox[2], ybox[3]]
    content += '\n'
    
    with open(label_save_path + img_name[:-4] + aug_type + ".txt", 'a') as f:
        for i in range(len(transformed["bboxes"])):
            fi.write(str(category_ids[i]) + " ")
            fi.write(str(bboxes[i][0]) + " ")
            fi.write(str(bboxes[i][1]) + " ")
            fi.write(str(bboxes[i][2]) + " ")
            fi.write(str(bboxes[i][3]) + "\n")
        
        
def get_crop_region(bbox, img_w, img_h):
    """ yolo bbox로 crop 영역을 계산
    
    Args:
        bbox: yolo bbox
        img_w: 이미지 너비
        img_h: 이미지 높이
    
    Returns:
        crop 영역 좌표
        
    """
    x, y, w, h = bbox
    w_half, h_half = w / 2, h / 2
    
    x_min, x_max = x - w_half, x + w_half
    y_min, y_max = y - h_half, y + h_half
    
    x_min = int(x_min * img_w)
    y_min = int(y_min * img_h)
    x_max = int(x_max * img_w)
    y_max = int(y_max * img_h)
    
    return [x_min, y_min, x_max, y_max]
        
    
def augment_pipeline(aug_type, image, bboxes, category_ids):
    """ 증강 파이프라인 설정 및 실행
      
    Args:
        aug_type: 증강 기법 종류
            crop: 크롭
            bright: 밝기
            zoom: 줌인, 줌아웃
            filp: 반전
        image: 증강 적용 대상 이미지
        bboxes: 증강 적용 대상 바운딩 박스들(2차원 리스트 형태)
        category_ids: 증강 적용 대상 클래스
            
    Returns:
        증강 적용 결과 transform
        
    """ 
    print(bboxes)
    
    if aug_type == "crop": # 물고기 bbox 크롭(crop)
        aug_type = "_crop"
        x_min, y_min, x_max, y_max = get_crop_region("""크롭영역=이미지외형바운딩박스""", img_width, img_height)
        transform = A.Compose(
            [A.Crop(x_min, y_min, x_max, y_max, p=1)],
            bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']), 
        )
        
    # 증강 실행
    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    return transformed


def augment_crop(image, bboxes, category_ids, crop_region):
    
    x_min, y_min, x_max, y_max = crop_region
    
    transform = A.Compose(
        [A.Crop(x_min, y_min, x_max, y_max, p=1)],
        bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
    )
        
    # 증강 실행
    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    return transformed
    
    
def get_box(data_img, data_ann, data_cls, data_dir):  
    """ 데이터셋에서 바운딩 박스를 가져와 변환 및 저장
    
    Args:
        data_img: 데이터셋의 이미지 정보
        data_ann: 데이터셋의 어노테이션 정보
        data_cls: 데이터셋의 카테고리 정보
        data_dir: 데이터셋의 디렉터리 경로
    
    Returns:
        
    """ 
    # target_num_list = [1, 2, 3, 6, 7, 8]
    target_num_list = [2, 3, 7, 8] # 0도 이미지 제외
    pbar = tqdm_notebook(total=None, desc=f"get_box: {data_dir}")
    
    for target_num in target_num_list:
        if target_num == 1:
            label_type = 'eye_0'
        
        elif target_num == 6:
            label_type = 'eyeless_0'
       
        elif target_num == 2 or target_num == 3:
            label_type = 'eye_45_90' 
        
        elif target_num == 7 or target_num == 8:
            label_type = 'eyeless_45_90'
        
        det_dir = os.path.join(data_dir, "object_detection")
        type_path = os.path.join(det_dir, label_type)
        
        label_path = os.path.join(type_path, 'labels')
        if not os.path.exists(label_path):  
            os.makedirs(label_path)
            
        image_path = os.path.join(type_path, 'images')
        if not os.path.exists(image_path):  
            os.makedirs(image_path)  
            
        # if target_num == 1 or target_num == 6:
        #     set_classes(type_path, class_side_0)
        # elif target_num == 2 or target_num == 3:
        #     set_classes(type_path, class_eye_45_90)
        # elif target_num == 7 or target_num == 8:
        #     set_classes(type_path, class_eyeless_45_90)
        
        for ann in data_ann:
            
            img_name = get_image(data_img, ann['image_id'], 'file_name')
            split_temp = img_name.split('_')[1]
            img_num = int(split_temp.split('.')[0].lstrip('0'))
            
            if target_num != img_num:
                continue
                                 
            code = get_class(data_cls, ann['category_id'], 'code')
            
            if code[:2] == 'PO': # 정상 라벨 제외
                continue
            elif code.find('OUY') != -1: #or code.find('OUN') != -1: # 외형 라벨, 크롭 기준
                # category = -1
                continue
            elif code not in class_code.keys():
                print(code)
                continue
            elif code[2:] not in class_eye_45_90 or code[2:] not in class_eyeless_45_90:
                continue
            else:
                if target_num == 1 or target_num == 6:
                    # category = class_side_0.index(code)
                    continue
                elif target_num == 2 or target_num == 3:
                    category = class_eye_45_90.index(code[2:])
                elif target_num == 7 or target_num == 8:
                    category = class_eyeless_45_90.index(code[2:])
            
            # cls_path = type_path + code[2:] + '/'
            # if not os.path.exists(cls_path):  
            #     os.makedirs(cls_path)
            # label_path = cls_path + 'labels/'
            # if not os.path.exists(label_path):  
            #     os.makedirs(label_path)
            # image_path = cls_path + 'images/'
            # if not os.path.exists(image_path):  
            #     os.makedirs(image_path)     
            
            
                
            size = get_image(data_img, ann['image_id'], 'size')
            yolo_bbox = coco_2_yolo(ann['bbox'], size)
            
            content = "{} {} {} {} {}\n".format(
                str(category), 
                str(yolo_bbox[0]),
                str(yolo_bbox[1]),
                str(yolo_bbox[2]),
                str(yolo_bbox[3])
            )
            # print(content)
            
            # .txt로 라벨 저장
            
            txt_path = os.path.join(label_path, img_name.split('.')[0] + ".txt")
            if ".." in txt_path:
                txt_path = txt_path.replace("..", ".")

            with open(txt_path, 'a') as f:
                f.write(content)
            
            if img_name.lower().endswith(".jpeg") or img_name.lower().endswith(".jpg"):
                img_name = img_name.split('.')[0] + '.JPG'
                
            # 학습에 사용할 이미지 복사
            source_path = os.path.join(data_dir, "01_image", img_name)
            target_path = os.path.join(image_path, img_name)
            shutil.copy(source_path, target_path)
            
            # 2. 이미지와 yolo 라벨을 증강 파이프라인에 통과시켜 저장
#             labels = set_boxes(img['file_name'], data_img, data_ann, data_cls)
            
#             if labels:
#                 save_yolo_bbox()

        # 외형 박스 추출 과정에서 생성된 외부 질병이 없는 데이터를 삭제(임시)
        for im in os.listdir(image_path):
            txt_file = os.path.join(label_path, im.split('.')[0] + '.txt')
            with open(txt_file, 'r') as file:
                file_contents = file.read().strip()
                lines = file_contents.split('\n')
                if len(lines) == 1 and lines[0].startswith("-1"):
                    # print(label_path + im.split('.')[0] + '.txt')
                    rm_img_file = os.path.join(image_path, im)
                    rm_txt_file = os.path.join(label_path, im.split('.')[0] + '.txt')
                    os.remove(rm_img_file)
                    os.remove(rm_txt_file)
          
        print(target_num)
        pbar.update(1)
        
    pbar.close()
    

def set_classes(path, class_list):
    """ classes.txt 파일을 생성
    
    Args:
        path: 저장 경로
        class_list: 모델별 클래스 목록
            
    Returns:
    
    """
    if not os.path.exists(path + "classes.txt"):   
        with open(path + "classes.txt", 'a') as f:
            for cl in class_list:
                f.write(cl + '\n')
              
            
def split_train_test(source_dir, test_rate):
    """ 학습셋과 테스트셋을 분할
    
    Args:
        source_dir: 원본 데이터셋 경로
        test_rate: 원본에서 테스트셋의 비율(20% = 0.2)
        
    Returns:
    
    """
    source = os.listdir(source_dir + 'images/')
    files = len(source)
    
    train_rate = 1. - test_rate
    
    test_cnt = int(files * test_rate)
    train_cnt = int(files * train_rate)
    
    test_set = random.sample(source, test_cnt) 
    
    if not os.path.exists(source_dir + 'test/images/'):  
        os.makedirs(source_dir + 'test/images/')
    if not os.path.exists(source_dir + 'test/labels/'):  
        os.makedirs(source_dir + 'test/labels/')
        
    for file in test_set: 
        shutil.move(source_dir + 'images/' + file, 
                    source_dir + 'test/images/' + file)
        shutil.move(source_dir + 'labels/' + file[:-4] + '.txt', 
                    source_dir + 'test/labels/' + file[:-4] + '.txt')
    
    source = os.listdir(source_dir + 'images/')
    
    if not os.path.exists(source_dir + 'train/images/'):  
        os.makedirs(source_dir + 'train/images/')
    if not os.path.exists(source_dir + 'train/labels/'):  
        os.makedirs(source_dir + 'train/labels/')
        
    for file in source:
        shutil.move(source_dir + 'images/' + file, 
                    source_dir + 'train/images/' + file)
        shutil.move(source_dir + 'labels/' + file[:-4] + '.txt', 
                    source_dir + 'train/labels/' + file[:-4] + '.txt')
    
    os.rmdir(source_dir + 'images/')
    os.rmdir(source_dir + 'labels/')
#     shutil.copy(source_dir + "classes.txt", source_dir + 'train/labels/' + "classes.txt")
#     shutil.copy(source_dir + "classes.txt", source_dir + 'test/labels/' + "classes.txt")
#     os.remove(source_dir + "classes.txt")
    
    return


def augmentation(dir):
    """ 증강 실행
    
    Args:
        dir: 증강할 데이터의 디렉터리(상위 디렉터리)
        
    Returns:
    
    """
    img_dir = dir + 'images/'
    label_dir = dir + 'labels/'
    
    label_file_list = os.listdir(label_dir)
    img_file_list = os.listdir(img_dir)
    
    aug_type = "crop"
    
    img_save_path = dir + aug_type + "/images/"
    label_save_path = dir + aug_type + "/labels/"
        
    if not os.path.exists(img_save_path):  
        os.makedirs(img_save_path)
    if not os.path.exists(label_save_path):  
        os.makedirs(label_save_path)
            
    
    for img in img_file_list:
        image = cv2.imread(img_dir + img, cv2.IMREAD_UNCHANGED)
        img_width = image.shape[1]
        img_height = image.shape[0]
        
        bboxes = []
        
        # 라벨 txt 파일 내용 "2차원 리스트로" 읽어오기
        with open(label_dir + img.split(".")[0] + ".txt", 'r') as f: 
            bbox = []
            category_ids = []
            crop_region = []
            while True:
                line = f.readline()
            
                if line == "":
                    break
                elif '\n' in line:
                    line.replace('\n', '')
                
                data = line.split(" ")
                
                if '-1' in data: # 외형 바운딩박스일 때
                    for d in data[1:]:
                        bbox.append(float(d))
                       
                    crop_region = get_crop_region(bbox, img_width, img_height)
                    
                else:
                
                    category_ids.append(int(data[0]))
                    for d in data[1:]:
                        bbox.append(float(d))
                
                    bboxes.append(bbox) 
                    
                bbox = []
                
                
        # print(img, " -> ", bboxes, " : ", category_ids)     
        
        if bboxes:
            transformed = augment_crop(image, bboxes, category_ids, crop_region)
    
        cv2.imwrite(img_save_path + img[:-4] + ".JPG", transformed["image"])
    
        with open(label_save_path + img[:-4] + ".txt", 'a') as fi:
            for i in range(len(transformed["bboxes"])):
                fi.write(str(category_ids[i]) + " ")
                fi.write(str(transformed["bboxes"][i][0]) + " ")
                fi.write(str(transformed["bboxes"][i][1]) + " ")
                fi.write(str(transformed["bboxes"][i][2]) + " ")
                fi.write(str(transformed["bboxes"][i][3]) + "\n")