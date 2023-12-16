import os
import json
import shutil

"""데이터셋의 최상위 폴더 경로"""
root_dir = "/media/lifeisbug/hdd/fish/fish_data/"
# root_dir_03 = "/home/lifeisbug/Downloads/ASD0327.v1i.coco/"  # 3월 데이터셋   
# root_dir_05 = root_dir + "flexink_05/"  # 5월 데이터셋
# root_dir_07 = "/home/lifeisbug/Downloads/flexink_07/"  # 7월 데이터셋
root_dir_08 = root_dir + "flexink_08/"  # 8월 데이터셋
root_dir_09 = root_dir + "flexink_09/"  # 9월 데이터셋



# data_08
# data_09
# data_08+09
# data_09 = dataset("20230908/")
# data_09.execute(inner, cls.get_organ) # inner or outer or all or key(11)
# data_09.execute(11, cls.get_organ)
# data_09.execute(key=11, func=cls.get_organ)
# data_09.execute(key=inner, func=cls.get_organ)

# class dataset:
#     def __init__(self, data_dir):
#         self.data_dir = root_dir + data_dir
#         self.imgs = 
#         self.anns = 
#         self.cats = 
#         self.dirs =
#         get_data()
    
#     def get_data(self,):
#         with open()
        
#     def execute(self, key, func):
        
        
    
    

"""플렉싱크에서 3월에 업로드한 라벨링 데이터에 대한 코드: 사용하지 않음"""
# data_03_train = dict()
# with open(root_dir_03 + "train/_annotations.coco.json", 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         data_03_train = data

# data_03_train['directory'] = root_dir_03 + "train/"

# data_03_valid = dict()
# with open(root_dir_03 + "valid/_annotations.coco.json", 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         data_03_valid = data

# data_03_valid['directory'] = root_dir_03 + "valid/"

# data_03_test = dict()
# with open(root_dir_03 + "test/_annotations.coco.json", 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         data_03_test = data

# data_03_test['directory'] = root_dir_03 + "test/"



keys = ['01', '02', '03', '06', '07', '08', 
        '11', '12', '13', '14', '15', '16', '17']
outer = keys[:6]
inner = keys[-7:]


"""플렉싱크에서 5월에 업로드한 라벨링 데이터에 대한 코드: 사용하지 않음"""

# 0516 데이터셋 불러오기
# dir_0516 = root_dir_05 + "20230516/" 
# img_dir_0516 = dir_0516 + "01_이미지데이터/"
# ann_dir_0516 = dir_0516 + "02_어노테이션데이터/"
# data_0516 = dict()
# for ann_dir in os.listdir(ann_dir_0516):
#     ann = ann_dir_0516 + ann_dir + "/segmentation.json"
    
#     with open(ann, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         data_0516[ann_dir[-2:]] = data

# data_0516['directory'] = img_dir_0516
        
# 0517 데이터셋 불러오기
# dir_0517 = root_dir_05 + "20230517/"
# img_dir_0517 = dir_0517 + "01_이미지데이터/"
# ann_dir_0517 = dir_0517 + "02_어노테이션데이터/"
# data_0517 = dict()
# for ann_dir in os.listdir(ann_dir_0517):
#     ann = ann_dir_0517 + ann_dir + "/segmentation.json"
    
#     with open(ann, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         data_0517[ann_dir[-2:]] = data
        
# data_0517['directory'] = img_dir_0517


from utils import is_multy_map

# def set_single_seg(ann_dir_05):
#     """ 어노테이션에서 멀티 세그먼테이션을 싱글 세그먼테이션으로 변경
#         리스트 내부의 리스트를 모두 삭제
#             Roboflow에서의 처리 방식을 따름: Roboflow는 맵 리스트 안의 리스트를 모두 삭제하고 숫자만 남겨 처리함
#         5월 데이터에 대해 적용(7월 데이터는 수정되었으므로 적용 불필요)
        
#     Args:
    
#     Returns:
        
#     """
#     for ann_dir in os.listdir(ann_dir_05):
#         dirs = ann_dir_05 + ann_dir
        
#         with open(dirs, 'r') as file:
#             data = json.load(file)
        
#         for anns in data['annotations']:
#             if anns['segmentation']:
#                 for ann in anns['segmentation']['counts']:
#                     while is_multy_map(ann):
#                         for a in ann:
#                             if isinstance(a, list):
#                                 del ann[ann.index(a)]
                        
#         for anns in data['annotations']:
#             if anns['segmentation']:
#                 for ann in anns['segmentation']['counts']:
#                     if is_multy_map(ann):
#                         print(ann)
        
#         # 수정된 내용을 .json 파일에 씀
#         with open(dirs, 'w') as file:
#             json.dump(data, file, indent=4)
        
        
# def get_imchang(todo_fold, todo_set):
#     """ 창현이가 직접 선택한 8 + 5월 데이터 목록 출력
    
#     Args:
        
#     Returns:
#         해당 파일명 리스트
        
#     """
#     todo_org = 'gill'
    
#     imchang_imgs = []
#     imchang_dir = "/media/lifeisbug/hdd/fish/fish_data/imchang/"
#     org_list = os.listdir(imchang_dir) # 장기별
#     for org in org_list:
#         classes = os.listdir(imchang_dir + '/' + todo_org + '/' + todo_fold + '/' + todo_set)
#         for c in classes:
#             images = os.listdir(imchang_dir + '/' + todo_org + '/' + todo_fold + '/' + todo_set + '/' + c)
#             for image in images:
#                 imchang_imgs.append(image[:13])
                            
#     yolo_imgs = []
#     source_dir = "/home/lifeisbug/JSON2YOLO/05+08/GILL_INS/labels/"
#     target_dir = "/home/lifeisbug/JSON2YOLO/05+08/GILL_INS/5-folds/"
#     for source in os.listdir(source_dir):
#         yolo_imgs.append(source)
                
#     for yolos in yolo_imgs:
#         if yolos[:13] in imchang_imgs:
#             # print(yolos)
#             shutil.copy(source_dir + yolos ,target_dir + '/' + todo_fold + '/' + todo_set + '/' + yolos)
        
    
"""플렉싱크에서 7월에 업로드한 라벨링 데이터에 대한 코드: 더 이상 사용하지 않음"""

# img_dir_07 = root_dir_07 + "01_image/"
# ann_dir_07 = root_dir_07 + "02_metadata/"

# # 7월 어노테이션 읽어오기
# data_07 = dict()
# for ann in os.listdir(ann_dir_07):
#     with open(ann_dir_07 + ann, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         data_07[ann[:2]] = data

# data_07['directory'] = img_dir_07


"""플렉싱크에서 8월에 업로드한 라벨링 데이터에 대한 코드"""
img_dir_08 = root_dir_08 + "01_image/"
ann_dir_08 = root_dir_08 + "02_metadata/"

# 8월 어노테이션 읽어오기
data_08 = dict()
for ann in os.listdir(ann_dir_08):
    if 'segmentation' in ann:
        with open(ann_dir_08 + ann, 'r') as f:
            data = json.load(f)
            data_08[ann[:2]] = data

data_08['directory'] = root_dir_08


"""플렉싱크에서 9월에 업로드한 라벨링 데이터에 대한 코드"""
img_dir_09 = root_dir_09 + "01_image/"
ann_dir_09 = root_dir_09 + "02_metadata/"

# 9월 어노테이션 읽어오기
data_09 = dict()
for ann in os.listdir(ann_dir_09):
    if 'segmentation' in ann:
        with open(ann_dir_09 + ann, 'r') as f:
            data = json.load(f)
            data_09[ann[:2]] = data

data_09['directory'] = root_dir_09


"""관련 함수"""

def collect_img(img_dir):
    """ 개체별 폴더로 구분된 이미지를 한 폴더로 모음
        기존의 코드는 이미지가 한 폴더에 모여있다 가정하고 작성됐기 때문에 호환을 위해 필요
        * 주의: 최초 한 번만 실행, 그 이후는 실행할 필요 없음
        
    Args:
        img_dir: 이미지 폴더 경로
        
    Returns:
        
    """
    error = set()
    for dirs in os.listdir(img_dir):
        for img in os.listdir(img_dir + '/' + dirs):
            if not img.startswith(dirs):
                error.add(dirs)
                fixed = img
                
                for i in data_09:
                    if i == 'directory':
                        continue
                        
                    for j in data_09[i]['images']:
                        json_fname = j['file_name']
                        if json_fname.startswith(dirs) :
                            print(f'이미지명: {img}, json 데이터: {json_fname}')
                            fixed = dirs[:3] + img[3:]
                            
                        elif json_fname == img:
                            print(f'폴더명: {dirs}, json 데이터: {json_fname}')
                            
                shutil.move(img_dir + dirs + '/' + img, img_dir + fixed)
                
            else:
                shutil.move(img_dir + dirs + '/' + img, img_dir + img)
                
            if len(os.listdir(img_dir + dirs)) == 0:
                os.rmdir(img_dir + dirs)
                
    # print(len(error))
    # print(error)
        

def copy_img(label_dir):
    """ 라벨 파일에 해당하는 이미지 파일을 복사해 옴
    
    Args:
        label_dir: 라벨 파일 경로
    
    Return:
    
    """
    target_dir = label_dir[:-7] + '/images/'
    
    for txt in os.listdir(label_dir):
        txt = txt[:-4]
        
        for img in os.listdir(img_dir_08):
            img = img[:-4]
            
            if img == txt:
                if (img + ".jpg") in os.listdir(img_dir_08):
                    shutil.copy(img_dir_08 + txt + ".jpg", target_dir + txt + ".jpg")
                elif (img + ".JPG") in os.listdir(img_dir_08):
                    shutil.copy(img_dir_08 + txt + ".JPG", target_dir + txt + ".JPG")
                
        for img in os.listdir(img_dir_09):
            img = img.split('.')[0]
            if img == txt:
                if (img + ".jpg") in os.listdir(img_dir_09):
                    shutil.copy(img_dir_09 + txt + ".jpg", target_dir + txt + ".jpg")
                elif (img + ".JPG") in os.listdir(img_dir_09):
                    shutil.copy(img_dir_09 + txt + ".JPG", target_dir + txt + ".JPG")
                elif (img + ".JPEG") in os.listdir(img_dir_09):
                    shutil.copy(img_dir_09 + txt + ".JPEG", target_dir + txt + ".JPEG")

#         for img in os.listdir(data_03_train['directory']):
#             img = img[:-4]
            
#             if img == txt:
#                 if (img + ".jpg") in os.listdir(data_03_train['directory']):
#                     shutil.copy(data_03_train['directory'] + txt + ".jpg", target_dir + txt + ".jpg")
#                 elif (img + ".JPG") in os.listdir(data_03_train['directory']):
#                     shutil.copy(data_03_train['directory'] + txt + ".JPG", target_dir + txt + ".JPG")
                    
#         for img in os.listdir(data_03_valid['directory']):
#             img = img[:-4]
            
#             if img == txt:
#                 if (img + ".jpg") in os.listdir(data_03_valid['directory']):
#                     shutil.copy(data_03_valid['directory'] + txt + ".jpg", target_dir + txt + ".jpg")
#                 elif (img + ".JPG") in os.listdir(data_03_valid['directory']):
#                     shutil.copy(data_03_valid['directory'] + txt + ".JPG", target_dir + txt + ".JPG")
        
#         for img in os.listdir(data_03_test['directory']):
#             img = img[:-4]
            
#             if img == txt:
#                 if (img + ".jpg") in os.listdir(data_03_test['directory']):
#                     shutil.copy(data_03_test['directory'] + txt + ".jpg", target_dir + txt + ".jpg")
#                 elif (img + ".JPG") in os.listdir(data_03_test['directory']):
#                     shutil.copy(data_03_test['directory'] + txt + ".JPG", target_dir + txt + ".JPG")
        
        
def execute_key(func, data, key):
    """ 기능을 구현한 함수를 데이터셋에서 특정 구분번호의 데이터에 대해 실행
    
    Args:
        func: 실행할 함수
        data: 실행 대상 데이터셋
        key: 이미지 구분 번호(01~17, '문자'로 0을 포함해 입력할 것)
    
    Returns:
    
    """
    func(data[key]['images'], 
         data[key]['annotations'], 
         data[key]['categories'], 
         data['directory'])
    
    
def execute_all(func, data):
    """ 기능을 구현한 함수를 데이터셋의 모든 데이터에 대해 실행
    
    Args:
        func: 실행할 함수
        data: 실행 대상 데이터셋
    
    Returns:
    
    """
    for key in keys:
        func(data[key]['images'], 
             data[key]['annotations'], 
             data[key]['categories'], 
             data['directory'])

    
def execute_inner(func, data):
    """ 기능을 구현한 함수를 데이터셋의 내부 데이터에 대해 실행
    
    Args:
        func: 실행할 함수
        data: 실행 대상 데이터셋
    
    Returns:
    
    """
    for key in inner:
        func(data[key]['images'], 
             data[key]['annotations'], 
             data[key]['categories'], 
             data['directory'])
    

def execute_outer(func, data):
    """ 기능을 구현한 함수를 데이터셋의 외부 데이터에 대해 실행
    
    Args:
        func: 실행할 함수
        data: 실행 대상 데이터셋
    
    Returns:
    
    """
    for key in outer:
        func(data[key]['images'], 
             data[key]['annotations'], 
             data[key]['categories'], 
             data['directory'])
    
    
"""플렉싱크에서 3월에 업로드한 라벨링 데이터(로보플로우 업로드)에 대한 코드: 더 이상 사용하지 않음"""

# import os
# import json

# set_to_num = {
#     0 : 'train', 
#     1 : 'valid', 
#     2 : 'test'
# }

# root_dir = "/home/lifeisbug/Downloads/ASD0327.v1i.coco/"
# train_dir = root_dir + "train/"
# valid_dir = root_dir + "valid/"
# test_dir = root_dir + "test/"

# # get json label
# # with open(train_dir + "_annotations.coco.json", 'r', encoding='utf-8') as f:
# #     train_data = json.load(f) 

# # with open(valid_dir + "_annotations.coco.json", 'r', encoding='utf-8') as f:
# #     valid_data = json.load(f) 
    
# # with open(test_dir + "_annotations.coco.json", 'r', encoding='utf-8') as f:
# #     test_data = json.load(f) 
    
# # pp(train_data['annotations'])

# # for k in train_data['annotations'][0].keys():
# #     print(k)

# def get_json_label(dir):
#     """ .json 라벨 파일을 읽어옴
    
#     Args:
#         dir: 파일 경로
    
#     Returns:
#         data: 읽어온 파일 데이터
    
#     """
#     with open(dir + "_annotations.coco.json", 'r', encoding='utf-8') as f:
#         data = json.load(f)
        
#     return data

# train_data = get_json_label(train_dir)
# train_ann = train_data['annotations']
# train_img = train_data['images']
# train_cls = train_data['categories']

# valid_data = get_json_label(valid_dir)
# valid_ann = valid_data['annotations']
# valid_img = valid_data['images']
# valid_cls = valid_data['categories']

# test_data = get_json_label(test_dir)
# test_ann = test_data['annotations']
# test_img = test_data['images']
# test_cls = test_data['categories']

# imgs = [train_img, valid_img, test_img]
# anns = [train_ann, valid_ann, test_ann]
# clss = [train_cls, valid_cls, test_cls]
# dirs = [train_dir, valid_dir, test_dir]

# def execute_func_for_3sets(func):
#     for set_ in set_to_num.keys():
#         func(imgs[set_], anns[set_], clss[set_], dirs[set_])