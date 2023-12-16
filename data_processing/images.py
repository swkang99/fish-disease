import os
import re
import cv2
from matplotlib import pyplot as plt

from directory import root_dir_08, root_dir_09
from utils import get_img_num
from dicts import class_code

def show_img(img, title):
    """ 이미지를 표시
    
    Args:
        img: 표시할 이미지
        title: 표시할 제목
        
    Returns:
    
    """
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
    

def save_img(img, file_name, root_name, idx=None, cls=None):
    """ 이미지를 클래스에 맞는 폴더에 저장
        폴더를 새로 생성하여 이미지를 저장하나, 이미 있는 폴더는 생성하지 않고 기존 폴더로 저장

    Args:
        img: 저장할 이미지
        file_name: 이미지 파일 이름
        root_name: 저장할 최상위 폴더명
        idx: 저장에 사용할 번호
            None일 경우, 파일명을 어노테이션 번호로 하지 않고 일괄 설정함
        cls: 저장할 폴더명(클래스 코드)
            None일 경우, 클래스 코드별로 폴더를 구분하지 않음(세그먼테이션 마스크 저장 시 사용)
        
    Returns:
    
    """
    img_num = get_img_num(file_name)
    dir_name = ''
    
    if img_num == '11' or img_num == '12':
        dir_name = '아가미, 간'
    elif img_num == '13':        
        dir_name = '아가미뚜껑'
    elif img_num == '14' or img_num == '15':
        dir_name = '장, 비장, 복수'
    elif img_num == '16' or img_num == '17':
        dir_name = '신장, 생식소'
        
    file_path = root_dir_09 + root_name + '/' + dir_name + '/'
    # file_path = root_dir_08 + root_name + '/'
    
    if cls is not None:
        
        file_path += cls + '(' + class_code[cls] + ')' + '/'
            
        if not os.path.exists(file_path): 
            os.makedirs(file_path)
                
        # if idx is not None:
        #     file_name += '_' + str(idx) + '.jpg'
        # else:
        #     file_name += 'cls.jpg'
            
        # 저장 코드는 if문 안에 있어야함
        # 밖에 있으면 예외처리된 파일을 저장하기 때문에 오류 발생 
        cv2.imwrite(file_path + file_name, img)
        
    else:
        if not os.path.exists(file_path): 
            os.makedirs(file_path)
            
        # file_name += '_mask.jpg'
        
        cv2.imwrite(file_path + file_name, img)
