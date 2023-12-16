import re
import os
import json
import random
import shutil

from dicts import class_code

def get_image(data_img, img_id, request):
    """ 이미지 정보를 찾아 반환함
    
    Args:
        data_img: 데이터셋의 이미지 정보 
        img_id: 이미지 id
        request: 요청의 종류
            file_name: 이미지 파일 이름
            size: 이미지 파일 사이즈(가로, 세로)
    Returns: 
        종류에 맞는 이미지 정보
        
    """ 
    for img in data_img:
        if img['id'] == img_id:
            if request == 'file_name':
                return img['file_name']
            elif request == 'size':
                return (img['width'], img['height'])

            
def get_class(data_cls, cls_id, request):
    """ 카테고리 id로부터 객체 클래스 코드 및 해당하는 질병 이름을 반환함
    
    Args:
        data_cls: 데이터셋의 카테고리 정보
        cls_id: 어노테이션의 카테고리 id
        request: 요청의 종류
            code: 객체 클래스 코드 그대로 반환
            name: 코드의 이름에 해당하는 질병 이름을 반환
            
    Returns: 
        일치하는 객체 클래스 코드 또는 이름, 예외발생 시 문구
        
    """
    result = ''
    
    for cls in data_cls:
        if cls['id'] == cls_id:
            result = cls['name']        
            
    if request == 'name': 
        if result not in class_code:
            return "class code not found: " + result
        else:
            return class_code[result]
    elif request == 'code':
        return result
    
    
def get_img_num(name):
    """ 이미지 파일 이름에서 이미지의 구분 번호를 추출
    
    Args:
        name: 이미지 파일 이름
        
    Returns:
        이미지의 구분 번호(01~17)
        
    """
    pattern = r"_(\d+)\."
    match = re.search(pattern, name)
    return match.group(1)
    

def count_class(data_img, data_ann, data_cls, data_dir):
    """ 클래스별 데이터 개수를 세어 출력
    
    Args:
        data_img: 데이터셋의 이미지 정보
        data_ann: 데이터셋의 어노테이션 정보
        data_cls: 데이터셋의 카테고리 정보
        data_dir: 데이터셋의 디렉터리 경로
        
    Returns:
    
    ** 현재 문제점: 3개 데이터셋에서의 통계가 아니라, 1개 데이터셋의 통계를 3번 출력함
    ** 개선 방안: 전체 ann에서 계산하기. 그러나 get_ 함수는 어떻게?
    ** 해결: 새로운 데이터셋은 구분번호별로 어노테이션 파일이 나뉘어 있으므로 구분이 변경됨
    """
    
    cls_num_dict = dict()
    cls_cnt_dict = dict()
                    
    for ann in data_ann:
        if not ann['bbox']:
            continue
            
        cls_code = get_class(data_cls, ann['category_id'], 'code')
        cls_num_dict[cls_code] = class_code[cls_code]
            
        if cls_code not in cls_cnt_dict.keys():
            cls_cnt_dict[cls_code] = 1
        else:
            cls_cnt_dict[cls_code] += 1
        
    for code in cls_cnt_dict.keys():
        print("{} : {}".format(class_code[code], cls_cnt_dict[code]))
    
    print("----------------------------------")
    

def is_multy_map(map):
    """ 세그먼테이션 맵이 Single인지 Multy인지 판단
        중첩된 리스트(nested list)일 때 Multy에 해당함
    
    Args:
        map: 세그먼테이션 맵
        
    Returns:
        True: Multy Segmentation
        False: Single Segmentation
        
    """
    for item in map:
        if isinstance(item, list):
            return True
    return False


def multy_seg_to_maps(seg):
    """ 멀티 세그먼테이션을 각 맵 리스트가 담긴 2차원 리스트로 변환
    
    Args:
        seg: 멀티 세그먼테이션 상태의 어노테이션 데이터
        
    Return:
        maps: 리스트로 된 맵들이 담긴 2차원 리스트
    
    """
    maps = []
    m = []
    for s in seg:
        if isinstance(s, int):
            m.append(s)
        else:
            maps.append(s)
            
    maps.insert(0, m)
    return maps
        
    
def split_data(dir, rate):
    """ 데이터셋을 정해진 비율로 분할
    
    Args:
        dir: 이미지 파일 경로
        rate: 분할 비율(테스트셋의 비율)
        
    Return:
    
    """
    source = os.listdir(dir) 
    train_dir = dir + 'train/images/'
    test_dir = dir + 'test/images/'
    label_dir = dir[:-7] + 'labels/'
    
    source_cnt = len(source)    
    test_cnt = int(source_cnt * rate)  
    test = random.sample(source, test_cnt)  
    
    for file in test:
        if not os.path.exists(test_dir): 
            os.makedirs(test_dir)
        shutil.move(dir + file, test_dir + file)
        
    move_label(test_dir, label_dir)
    shutil.move(test_dir[:-7], test_dir[:-19])
    
    train = os.listdir(dir)
    for file in train:
        if not os.path.exists(train_dir): 
            os.makedirs(train_dir)
        shutil.move(dir + file, train_dir + file)
    
    move_label(train_dir, label_dir) 
    shutil.move(train_dir[:-7], train_dir[:-20])
    
    os.rmdir(dir)
    os.rmdir(label_dir)
    return


def move_label(img_dir, txt_dir):
    """ 데이터(이미지)에 맞는 라벨 파일을 옮김
        * YOLO 형식의 라벨에 대해 작동함
    
    Args:
        img_dir: 이미지 파일 경로
        txt_dir: 라벨(텍스트) 파일 경로

    Return:
    
    """
    # txt 파일을 옮길 목적지
    target_dir = img_dir[:-7] + 'labels/'
    os.makedirs(target_dir)
    
    img_list = os.listdir(img_dir)
    txt_list = os.listdir(txt_dir)

    for txt in txt_list:
        txt = txt[:-4]
        
        for img in img_list:
            img = img[:-4]
            
            if img == txt:
                
                shutil.move(txt_dir + txt + ".txt", target_dir + txt + ".txt")
                
                if (img + ".jpg") in img_list:
                    img_list.remove(img + ".jpg")
                elif (img + ".JPG") in img_list:
                    img_list.remove(img + ".JPG")
                    
                break
    
    
def set_class_part_symptom(ann_seg_dir):
    """ 내부 장기 라벨을 증상 + 부위 라벨로 수정
    
    Args: 
        ann_seg_dir: 어노테이션 파일 경로
    
    Return:
    
    """
    for ann_dir in os.listdir(ann_seg_dir):
        dirs = ann_seg_dir + ann_dir
        
        if 'json' not in dirs:
            continue
            
        with open(dirs, 'r') as file:
            data = json.load(file)
        
        if 'segmentation' not in ann_dir:
            continue
                
        for cats in data['categories']:
            if '11' in ann_dir or '12' in ann_dir:
                if 'gill' in ann_dir:     # 아가미 모델
                    cats['supercategory'] = cats['name'][2:]
                    if cats['supercategory'] == 'GIN':
                        id = 0
                        cats['name'] = 'GILL_NORMAL' # 0 : 정상 아가미
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                    
                    elif cats['supercategory'] == 'GIP':
                        id = 1
                        cats['name'] = 'GILL_PANMY' # 1 : 아가미 빈혈
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                        
                    else:
                        data['annotations'] = remove_ann(data['annotations'], cats['id'])
                
                elif 'liver' in ann_dir:   # 간 모델
                    cats['supercategory'] = cats['name'][2:] 
                    if cats['supercategory'] == 'LIN':
                        id = 0
                        cats['name'] = 'LIVER_NORMAL' # 0 : 정상 간
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id

                    elif cats['supercategory'] == 'LIC':
                        id = 1
                        cats['name'] = 'LIVER_CONGESTION' # 1 : 간 염증
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id

                    elif cats['supercategory'] == 'LII':
                        id = 2
                        cats['name'] = 'LIVER_INFLAMM' # 2 : 간 울혈
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                    
                    elif cats['supercategory'] == 'LIP':
                        id = 3
                        cats['name'] = 'LIVER_PANMY' # 3 : 간 빈혈
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                     
                    else:
                        data['annotations'] = remove_ann(data['annotations'], cats['id'])
        
            elif '13' in ann_dir:
                cats['supercategory'] = cats['name'][2:]
                if cats['supercategory'] == 'GCN':
                    id = 0
                    cats['name'] = 'GILL_COVER_NORMAL' # 0 : 정상 아가미뚜껑
                    data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                    cats['id'] = id
                    
                elif cats['supercategory'] == 'GCH':
                    id = 1
                    cats['name'] = 'GILL_COVER_HEMORR' # 1 : 아가미뚜껑 안쪽 출혈
                    data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                    cats['id'] = id
                    
                else:
                    data['annotations'] = remove_ann(data['annotations'], cats['id'])
                    
            elif '14' in ann_dir or '15' in ann_dir:
                if 'intestine' in ann_dir:
                    cats['supercategory'] = cats['name'][2:]
                    if cats['supercategory'] == 'INN':
                        id = 0
                        cats['name'] = 'INTESTINE_NORMAL' # 0 : 정상 장
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                    
                    elif cats['supercategory'] == 'INI':
                        id = 1
                        cats['name'] = 'INTESTINE_COMPOUND' # 1 : 장 염증(INFLAMM)
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                    
                    elif cats['supercategory'] == 'INH':
                        id = 1
                        cats['name'] = 'INTESTINE_COMPOUND' # 2 : 장 출혈(HEMORR)
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                    
                    else:
                        data['annotations'] = remove_ann(data['annotations'], cats['id'])
                    
                elif 'ascites' in ann_dir:
                    cats['supercategory'] = cats['name'][2:]
                    if cats['supercategory'] == 'ASN':
                        id = 0
                        cats['name'] = 'ASCITES_NORMAL' # 0 : 정상 복수
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                    
                    elif cats['supercategory'] == 'ASQ':
                        id = 1
                        cats['name'] = 'ASCITES_OPAQUE' # 1 : 탁한 복수
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                    
                    elif cats['supercategory'] == 'ASC':
                        id = 2
                        cats['name'] = 'ASCITES_CLEAN' # 2 : 맑은 복수
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                        
                    elif cats['supercategory'] == 'ASH':
                        id = 1
                        cats['name'] = 'ASCITES_OPAQUE' # 3 : 출혈성 복수(ASCITES_HEMORR)이나, 탁한 복수로 통합하여 학습
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                
                    else:
                        data['annotations'] = remove_ann(data['annotations'], cats['id'])
        
        cat_to_delete = []
        for cat in data['categories']:
            if len(cat['name']) == 5 and cat['name'] != 'LIVER':
                cat_to_delete.append(cat)
            
        for cat in cat_to_delete:
            data['categories'].remove(cat)           
                
        # 수정된 내용을 .json 파일에 씀
        with open(dirs, 'w') as file:
            json.dump(data, file, indent=4)
    

def set_class_part(ann_org_dir):
    """ 내부 장기 라벨을 부위 라벨로 수정
    
    Args: 
        ann_org_dir: 어노테이션 파일 경로
    
    Return:
    
    """
    for ann_dir in os.listdir(ann_org_dir):
        dirs = ann_org_dir + ann_dir
        if 'segmentation' not in ann_dir:                
            continue
            
            
        with open(dirs, 'r') as file:
            data = json.load(file)
        
        
        for cats in data['categories']:
            # if '11' in ann_dir or '12' in ann_dir:
                if 'gill' in ann_dir:
                    cats['supercategory'] = cats['name'][2:4]
                    if cats['supercategory'] == 'GI':
                        id = 0
                        cats['name'] = 'GILL' # 아가미
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                    else:
                        data['annotations'] = remove_ann(data['annotations'], cats['id'])
                        
                elif 'liver' in ann_dir:
                    cats['supercategory'] = cats['name'][2:4]
                    if cats['supercategory'] == 'LI':
                        id = 0
                        cats['name'] = 'LIVER' # 간
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                    else:
                        data['annotations'] = remove_ann(data['annotations'], cats['id'])
            
                if '13' in ann_dir:
                    cats['supercategory'] = cats['name'][2:4]
                    if cats['supercategory'] == 'GC':
                        id = 0
                        cats['name'] = 'GILL_COVER' # 아가미뚜껑
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                    
                    else:
                        data['annotations'] = remove_ann(data['annotations'], cats['id'])
                    
            # elif '14' in ann_dir or '15' in ann_dir:
                if 'intestine' in ann_dir: 
                    cats['supercategory'] = cats['name'][2:4]
                    if cats['supercategory'] == 'IN':
                        id = 0
                        cats['name'] = 'INTESTINE' # 장
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                    
                    else:
                        data['annotations'] = remove_ann(data['annotations'], cats['id'])
                
                elif 'ascites' in ann_dir:
                    cats['supercategory'] = cats['name'][2:4]
                    if cats['supercategory'] == 'AS':
                        id = 0
                        cats['name'] = 'ASCITES' # 복수
                        data['annotations'] = set_new_category_id(data['annotations'], cats['id'], id)
                        cats['id'] = id
                    
                    else:
                        data['annotations'] = remove_ann(data['annotations'], cats['id'])
                    
#             elif ann_dir == 'anno_16' or ann_dir == 'anno_17':
#                 cats['supercategory'] = cats['name'][2:4]
#                 if cats['supercategory'] == 'AS':
#                     cats['name'] = 'ASCITES' # 복수
#                     data['annotations'] = set_new_category_id(data['annotations'], cats['id'], 0)
#                     cats['id'] = 0
                    
#                 else:
#                     data['annotations'] = remove_ann(data['annotations'], cats['id'])
        
        cat_to_delete = []
        for cat in data['categories']:
            if len(cat['name']) == 5 and cat['name'] != 'LIVER':
                cat_to_delete.append(cat)
            
        for cat in cat_to_delete:
            data['categories'].remove(cat)     
            
        # 수정된 내용을 .json 파일에 씀
        with open(dirs, 'w') as file:
            json.dump(data, file, indent=4)
            

def set_new_category_id(data_ann, old_id, new_id):
    """ 카테고리 id를 수정
    
    Args:
        data_ann: 어노테이션 데이터
        old_id: 기존 id
        new_id: 새 id
        
    Return:
        수정된 어노테이션 데이터
        
    """
    for ann in data_ann:
        if ann['category_id'] == old_id:
            ann['category_id'] = new_id
            
    return data_ann


def remove_ann(data_ann, target_id):
    """ 특정 카테고리 id의 어노테이션을 삭제
    
    Args:
        data_ann: 어노테이션 데이터
        target_id: 삭제 대상인 어노테이션의 카테고리 id
        
    Return:
        수정된 어노테이션 데이터
        
    """
    ann_to_delete = []
    for ann in data_ann:
        if ann['category_id'] == target_id:
            ann_to_delete.append(ann)
            
    for ann in ann_to_delete:
        data_ann.remove(ann)
        
    return data_ann