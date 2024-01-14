from enum import Enum

# 아가미 증상
class GillSymptom(Enum):
    PANMY = 0  # 빈혈
    NORMAL = 1 # 정상

# 간 증상
class LiverSymptom(Enum):
    CONGEST = 0 # 울혈
    NORMAL = 1  # 정상
    INFLAM = 2  # 염증
    PANMY = 3   # 빈혈

# 아가미뚜껑 증상
class GillCoverSymptom(Enum):
    NORMAL = 0 # 정상
    HEMORR = 1 # 염증 및 출혈

# 장 증상
class IntestineSymptom(Enum):
    COMPOUND = 0 # 염증 및 출혈
    NORMAL = 1   # 정상

# 복수 증상
class AscitesSymptom(Enum):
    NORMAL = 0 # 정상
    HEMORR = 1 # 출혈성(탁한) 복수
    CLEAN = 2  # 맑은 복수

# 외부 유안측 증상
class FRSymptom(Enum):
    DYH = 0 # 체표 출혈
    # DYU = 1 # 체표 궤양
    # DYA = 2 # 체표 근육 출혈
    FDH = 1 # 등지느러미 출혈
    FAH = 2 # 뒷지느러미 출혈
    FCH = 3 # 꼬리지느러미 출혈
    # MOU = 6 # 주둥이 궤양

# 외부 무안측 증상
class RSSymptom(Enum):
    DYH = 0 # 체표 출혈
    # DYU = 1 # 체표 궤양
    # DYA = 2 # 체표 근육 출혈
    FDH = 1 # 등지느러미 출혈
    FAH = 2 # 뒷지느러미 출혈
    FCH = 3 # 꼬리지느러미 출혈
    # MOU = 6 # 주둥이 궤양