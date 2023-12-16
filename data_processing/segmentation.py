import cv2
import numpy as np

from images import show_img, save_img
from utils import get_image, get_class, is_multy_map, multy_seg_to_maps
from dicts import class_code

def map_with_img(img, map, cls, dir):
    """ 이미지에 세그먼테이션 맵을 그림
    
    Args:
        img: 이미지 파일 이름
        map: 세그먼테이션 맵(2차원 리스트)
        cls: 질병 이름
        dir: 데이터셋의 이미지 디렉터리 경로
        
    Returns:
    
    """
    # 원본 이미지 표시
    # origin = cv2.imread(dir + img, cv2.IMREAD_UNCHANGED)
    # show_img(origin, "original image")
    
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
      
    areas = cvt_map(map)
    
    # 모든 map 시각화
    for area in areas: 
        origin = cv2.imread(dir + img, cv2.IMREAD_UNCHANGED)
        visualized = cv2.polylines(origin, [area], True, green, 2)
        show_img(visualized, img)
        print("{}({})".format(cls, class_code[cls]))
        
    
def cvt_map(maps):
    """ 세그먼테이션 맵을 cv2.polylines 함수에 입력할 수 있도록 변환함

    Args:
        maps: 세그먼테이션 맵(2차원 리스트)
        
    Returns:
        변환된 세그먼테이션 맵
        
    """
    results = []
    
    for m in maps:
    
        result = []
        xy = []
        polygons = m

        for p in polygons:
        
            xy.append(p)
            if len(xy) >= 2:
                result.append(xy)
                xy = []   
                
        results.append(np.array(result, dtype=np.int32))
        
    return results

    
def set_polygons(img_name, data_img, data_ann, data_cls):
    """ 어노테이션에서 마스크를 만들 폴리곤과 마스크 색깔을 설정
    
    Args:
        img_name: 원본 이미지 파일 이름
        data_img: 데이터셋의 이미지 정보
        data_ann: 데이터셋의 어노테이션 정보
        data_cls: 데이터셋의 카테고리 정보
    
    Returns:
        폴리곤 좌표, 마스크 색상으로 이루어진 튜플의 리스트
    
    """
    polygons = []
    
    for ann in data_ann:
        if (get_image(data_img, ann['image_id'], 'file_name') == img_name and \
            ann['segmentation']):
            map = cvt_map(ann['segmentation'])
            code = get_class(data_cls, ann['category_id'], 'code')
            
#             if code in class_code.keys():
#                 # color = class_color[code] # 마스크 색을 클래스 고유 색으로
#                 color = (255, 255, 255)
#                 # save_mask_to_csv(img_name, ann['id'])
            
#             else:
#                 continue

            if ann['category_id'] == 0:
                color = (0, 0, 255) # Red
            elif ann['category_id'] == 1:
                color = (255, 0, 0) # Blue
            else:
                print("Class Not Found")
                continue

            # color = (255, 255, 255) # white mask for grayscale
                
            polygons.append((map, color))
            
    return polygons


def save_mask_to_csv(img_name, ann_id):
    """ 세그먼테이션 마스크 이미지 정보를 .csv 파일로 저장
    
    Args:
        img_name: 원본 이미지 파일 이름
        ann_id: 어노테이션 id
    
    Returns:
    
    """
    
    with open("segmentation_mask.csv", 'a') as f:
        f.write(img_name + ',' + str(ann_id) + '\n')
        
        
def masking_map(img, polygons, dir):
    """  이미지의 세그먼테이션 마스크를 만들어 별도 이미지로 저장
    
    Args:
        img: 이미지 파일 이름
        polygons: 마스크로 그릴 폴리곤 리스트
        dir: 데이터셋의 이미지 폴더 경로
        
    Returns:
    
    """
    # 이미지 파일 불러오기
    image = cv2.imread(dir + img, cv2.IMREAD_UNCHANGED)
    show_img(image, "original image")
    
    # 다각형 모양의 마스크 이미지 생성
    mask = np.zeros_like(image) # 원본 이미지와 같은 크기의 빈 이미지 생성
    for poly in polygons:
        
        pts = poly[0]
        color = poly[1]
        cv2.fillPoly(mask, pts, color) # 다각형을 채우기
    
    # 그레이스케일로 변환
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    show_img(mask, "mask")
    
    # 이미지 저장
    save_img(mask, img, 'segmentation_mask_color')
    
    
def mask_from_image(data_img, data_ann, data_cls, data_dir):  
    """ 원본 이미지와 세그먼테이션 맵으로 학습에 사용할 마스크 이미지 생성
    
    Args:
        data_img: 데이터셋의 이미지 정보
        data_ann: 데이터셋의 어노테이션 정보
        data_cls: 데이터셋의 카테고리 정보
        data_dir: 데이터셋의 이미지 폴더 경로
    
    Returns:
        
    """
    for img in data_img:
        polygons = set_polygons(img['file_name'], data_img, data_ann, data_cls)
        if polygons:
            mask = masking_map(img['file_name'], polygons, data_dir)
    
    
def visualize_map(data_img, data_ann, data_cls, data_dir):  
    """ 원본 이미지에 세그먼테이션 맵 시각화
    
    Args:
        data_img: 데이터셋의 이미지 정보
        data_ann: 데이터셋의 어노테이션 정보
        data_cls: 데이터셋의 카테고리 정보
        data_dir: 데이터셋의 이미지 폴더 경로
    
    Returns:
        
    """ 
    for ann in data_ann:
        ann_maps = ann['segmentation']

        for ann_map in ann_maps:
            map_with_img(get_image(data_img, ann['image_id'], 'file_name'),
                        [ann_map],
                        get_class(data_cls, ann['category_id'], 'code'),
                        data_dir)