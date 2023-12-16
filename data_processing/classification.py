import cv2
import numpy as np

from utils import get_image, get_class, is_multy_map, multy_seg_to_maps
from images import show_img, save_img
from segmentation import cvt_map


def crop(image, points, img):
    """ 이미지의 지정 영역을 자름
    
    Args:
        image: 이미지
        points: 잘라낼 영역
        
    Returns:
        잘라낸 이미지
        
    """
    x, y, w, h = cv2.boundingRect(points)
    
    cropped_image = image[y:y+h, x:x+w]
    
    mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points  - (x, y)], 255)
    
    masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
    
    # show_img(masked_image, "masked")
    # print(masked_image.shape[1], masked_image.shape[0])
    return masked_image
    

def set_scale(masked_image, target_scale):
    """ 잘라낸 장기 이미지를 목표 크기로 변경
    
    Args:
        masked_image: 잘라낸 이미지
        target_scale: 결과 이미지의 크기(너비, 높이)
            
    Returns:
        변경된 크기의 이미지
        
    """
    resized_image = cv2.resize(masked_image, target_scale)
    return resized_image
               
    
def down_scale(masked_image, target_scale):
    """ 잘라낸 장기 이미지 크기(너비 또는 높이)가 목표 크기 이하가 되도록 축소
    
    Args:
        masked_image: 잘라낸 이미지
        target_scale: 결과 이미지의 크기(너비, 높이)
            
    Returns:
        축소된 이미지
        
    """
    width = masked_image.shape[1]
    height = masked_image.shape[0]
    
    target_width = target_scale[0]
    target_height = target_scale[1]
    
    ratio = min(target_width / width, target_height / height)
    
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    resized_image = cv2.resize(masked_image, (new_width, new_height))
    
    # show_img(resized_image, "resized")
    # print(resized_image.shape[1], resized_image.shape[0])
    return resized_image


def border(resized_image, target_scale):
    """ 축소된 이미지에 검은 부분을 채워 목표 사이즈로 수정
    
    Args:
        resized_image: 축소된 이미지
        target_scale: 목표 사이즈(결과 이미지의 크기(너비, 높이))
            
    Returns:
        채워진 이미지
        
    """
    target_width = target_scale[0]
    target_height = target_scale[1]
    
    width_diff = target_width - resized_image.shape[1]
    height_diff = target_height - resized_image.shape[0]
    
    border_color = (0, 0, 0) # Black
    border_image = cv2.copyMakeBorder(resized_image, 
                                      0, 
                                      height_diff, 
                                      0, 
                                      width_diff, 
                                      cv2.BORDER_CONSTANT, 
                                      value=border_color)
    
    # show_img(border_image, "bordered")
    # print(border_image.shape[1], border_image.shape[0])
    return border_image
    
    
def cut_map(img, maps, cls, dir, idx, target_scale, options):
    """ 이미지의 맵 부분을 잘라서 파일로 저장
    
    Args:
        img: 이미지 파일 이름
        maps: 세그먼테이션 맵(2차원 리스트)
        cls: 객체 클래스 코드
        dir: 데이터셋의 이미지 폴더 경로
        idx: 어노테이션 id(저장에 사용할 번호)
        target_scale: 결과 이미지의 크기(너비, 높이)
        options: 처리 종류 지정
            crop : 원본 이미지로부터 장기 이미지만 자르기
            resize : 잘라낸 장기 이미지 크기(너비 또는 높이)가 목표 크기 이하가 되도록 축소
            border: 축소된 이미지에 검은 부분을 채워 목표 크기로 수정
            
    Returns:
    
    """
    image = cv2.imread(dir + img, cv2.IMREAD_UNCHANGED)
    map = cvt_map(maps)
    
    for points in map:
        if not options: # 원본 이미지를 분류셋 구조로 저장 : 처리 종류를 지정하지 않았을 때 사용
            save_img(image, img, 'original', idx, cls) 
        
        else:
            if 'crop' in options:
                result = crop(image, points, img)

            if 'set_scale' in options:
                result = set_scale(result, target_scale)
            elif 'down_scale' in options:
                result = down_scale(result, target_scale)

            if 'border' in options:
                result = border(result, target_scale)
        
            save_img(result, img, 'classification_original_border', idx, cls)


def get_organ(data_img, data_ann, data_cls, data_dir):
    """ 데이터셋의 원본 이미지로부터 장기 이미지를 추출
    
    Args:
        data_img: 데이터셋의 이미지 정보
        data_ann: 데이터셋의 어노테이션 정보
        data_cls: 데이터셋의 카테고리 정보
        data_dir: 데이터셋의 이미지 폴더 경로
    
    Returns:
        
    """ 
    target_size = (256, 256) # 원하는 출력 이미지 크기
    # options = ['crop', 'set_scale'] # 원하는 처리만 지정
    options = ['crop', 'down_scale', 'border']
    
    for ann in data_ann:
        ann_maps = ann['segmentation']
            
        for ann_map in ann_maps:
            img_name = get_image(data_img, ann['image_id'], 'file_name')
            if img_name[0] == 'W':
                continue
                
            cut_map(img_name, 
                    [ann_map], 
                    get_class(data_cls, ann['category_id'], 'code'), 
                    data_dir,
                    ann['id'],
                    target_size,
                    options)
