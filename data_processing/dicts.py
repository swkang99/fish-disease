import random

def random_color():
    """ 무작위 RGB 색상을 생성
    
    Args:

    Returns:
        RGB 색상값 튜플

    """
    r = lambda: random.randint(0, 255)
    return (r(), r(), r())
                                        
def set_class_color():
    """ 클래스 코드별로 무작위 색상을 부여

    Args:

    Returns:
        { 코드 : (색상) } 딕셔너리
    
    """
    class_color = {}                                        
    for code in class_code.keys():
        class_color[code] = random_color()
    return class_color

def get_class_color(class_code):
    """ 클래스 코드에 해당하는 색상을 조사
    
    Args:
        class_code: 클래스 코드

    Returns: 
        RGB 색상값 튜플
    """
    return class_color[class_code]

# class_color = set_class_color()
# print(get_class_color('EPDYR'))


"""object class code for detections"""
class_side_0 = [ # both eye and eyeless
    'VHBEB',
    'VHNAP',
    'SPBEB',
    'POBEN',
    'PONAN'
]

class_eye_45_90 = [
    'DYH', # 체표 출혈
    'FDH', # 등지느러미 출혈
    'FAH', # 뒷지느러미 출혈
    'FCH', # 꼬리지느러미 출혈
    # 'DYA' # 체표 근육 출혈
    #'DYU', # 체표 궤양
    #'MOU'  # 주둥이 궤양
]

class_eyeless_45_90 = [
    'DYH', # 체표 출혈
    'FDH', # 등지느러미 출혈
    'FAH', # 뒷지느러미 출혈
    'FCH', # 꼬리지느러미 출혈
    # 'DYA' # 체표 근육 출혈
    #'DYU', # 체표 궤양
    #'MOU',  # 주둥이 궤양
]

"""5월 데이터 object class code"""
class_code = {
    'EPDYR': '에드워드병 체표 발적',
    'EPDYN': '에드워드병 체표 괴사',
    'EPDYA': '에드워드병 체표 근육출혈',
    'EPBEB': '에드워드병 복부 팽만',
    'EPNAI': '에드워드병 항문 염증',
    'EPNAP': '에드워드병 항문 탈장',
    'EPASH': '에드워드병 출혈성 복수',
    'EPLIF': '에드워드병 간 섬유화',
    'EPLIA': '에드워드병 간 농양',
    'EPLIP': '에드워드병 간 빈혈',
    'EPKIE': '에드워드병 신장 비대',
    'EPKIA': '에드워드병 신장 농양',
    'EPKIN': '에드워드병 신장 결절',
    'EPKII': '에드워드병 신장 염증',
    'EPSPE': '에드워드병 비장 비대',
    'EPINI': '에드워드병 장 염증',
    'EPINH': '에드워드병 장 출혈',
    'EPINT': '에드워드병 장벽얇아짐',
    'EPOUY': '에드워드병 외형',
    'VIDYR': '비브리오증 체표 발적',
    'VIDYU': '비브리오증 체표 궤양',
    'VIDYH': '비브리오증 체표 출혈',
    'VIDYN': '비브리오증 체표 괴사',
    'VIDYA': '비브리오증 체표 근육 출혈',
    'VIMOU': '비브리오증 주둥이 궤양',
    'VIGIP': '비브리오증 아가미 빈혈',
    'VIFDS': '비브리오증 등지느러미 갈라짐',
    'VIFAS': '비브리오증 뒷지느러미 갈라짐',
    'VIFCS': '비브리오증 꼬리지느러미 갈라짐',
    'VIFDH': '비브리오증 등지느러미 출혈',
    'VIFAH': '비브리오증 뒷지느러미 출혈',
    'VIFCH': '비브리오증 꼬리지느러미 출혈',
    'VIFDN': '비브리오증 등지느러미 괴사',
    'VIFAN': '비브리오증 뒷지느러미 괴사',
    'VIFCN': '비브리오증 꼬리지느러미 괴사',
    'VILIC': '비브리오증 간 울혈',
    'VILII': '비브리오증 간 염증',
    'VIKIE': '비브리오증 신장 비대',
    'VISPE': '비브리오증 비장 비대',
    'VIOUY': '비브리오증 외형',
    'SPDYH': '연쇄구균병 체표 출혈',
    'SPDYF': '연쇄구균병 체표 미병부 출혈',
    'SPBEB': '연쇄구균병 복부 팽만',
    'SPEYB': '연쇄구균병 눈 돌출',
    'SPEYH': '연쇄구균병 눈 출혈',
    'SPGCH': '연쇄구균병 아가미뚜껑 안쪽 출혈',
    'SPGIP': '연쇄구균병 아가미 빈혈',
    'SPGIN': '연쇄구균병 아가미 괴사',
    'SPASQ': '연쇄구균병 탁한 복수',
    'SPASH': '연쇄구균병 출혈성 복수',
    'SPFDH': '연쇄구균병 등지느러미 출혈',
    'SPFAH': '연쇄구균병 뒷지느러미 출혈',
    'SPFCH': '연쇄구균병 꼬리지느러미 출혈',
    'SPNAP': '연쇄구균병 항문 탈장',
    'SPLIC': '연쇄구균병 간 울혈',
    'SPLIE': '연쇄구균병 간 비대',
    'SPLIP': '연쇄구균병 간 빈혈',
    'SPKIE': '연쇄구균병 신장 비대',
    'SPSPE': '연쇄구균병 비장 비대',
    'SPINI': '연쇄구균병 장 염증',
    'SPINH': '연쇄구균병 장 출혈',
    'SPINT': '연쇄구균병 장벽얇아짐',
    'SPREH': '연쇄구균병 생식소 출혈',
    'SPRED': '연쇄구균병 국소적흑화',
    'SPOUY': '연쇄구균병 외형',
    'TMDYS': '활주세균병 체표 번짐괴사',
    'TMMOU': '활주세균병 주둥이궤양',
    'TMGIP': '활주세균병 아가미 빈혈',
    'TMGIL': '활주세균병 아가미 결손',
    'TMFDL': '활주세균병 등지느러미 결손',
    'TMFAL': '활주세균병 뒷지느러미 결손',
    'TMFCL': '활주세균병 꼬리지느러미 결손',
    'TMFDN': '활주세균병 등지느러미 괴사',
    'TMFAN': '활주세균병 뒷지느러미 괴사',
    'TMFCN': '활주세균병 꼬리지느러미 괴사',
    'TMOUY': '활주세균병 외형',
    'ELDYE': '여윔증 체표 여윔',
    'ELDYD': '여윔증 체표 두부함몰',
    'ELLID': '여윔증 간 흑적색',
    'ELOUY': '여윔증 외형',
    'MADYU': '스쿠티카병 체표 근육노출궤양',
    'MAMOU': '스쿠티카병 주둥이 궤양',
    'MAFDL': '스쿠티카병 등지느러미 결손',
    'MAFAL': '스쿠티카병 뒷지느러미 결손',
    'MAFCL': '스쿠티카병 꼬리지느러미 결손',
    'MAFDH': '스쿠티카병 등지느러미 출혈',
    'MAFAH': '스쿠티카병 뒷지느러미 출혈',
    'MAFCH': '스쿠티카병 꼬리지느러미 출혈',
    'MAFDB': '스쿠티카병 등지느러미 기부괴사',
    'MAFAB': '스쿠티카병 뒷지느러미 기부괴사',
    'MAFCB': '스쿠티카병 꼬리지느러미 기부괴사',
    'MAGIP': '스쿠티카병 아가미 빈혈',
    'MAGIN': '스쿠티카병 아가미 괴사',
    'MAOUY': '스쿠티카병 외형',
    'VHDYR': '바이러스성출혈성패혈증 체표 발적',
    'VHDYA': '바이러스성출혈성패혈증 체표 근육내출혈',
    'VHBEB': '바이러스성출혈성패혈증 복부 팽만',
    'VHGIP': '바이러스성출혈성패혈증 아가미 빈혈',
    'VHNAP': '바이러스성출혈성패혈증 항문 탈장',
    'VHASC': '바이러스성출혈성패혈증 맑은 복수',
    'VHLIC': '바이러스성출혈성패혈증 간 울혈',
    'VHKIE': '바이러스성출혈성패혈증 신장 비대',
    'VHSPE': '바이러스성출혈성패혈증 비장 비대',
    'VHOUY': '바이러스성출혈성패혈증 외형',
    'POOUN': '정상 외형',
    'PODYN': '정상 체표',
    'PONAN': '정상 항문',
    'POFDN': '정상 등지느러미',
    'POFAN': '정상 뒷지느러미',
    'POFCN': '정상 꼬리지느러미',
    'POEYN': '정상 눈',
    'POMON': '정상 입',
    'POBEN': '정상 복부',
    'POGCN': '정상 아가미뚜껑',
    'POGIN': '정상 아가미',
    'POLIN': '정상 간',
    'POINN': '정상 장',
    'POSPN': '정상 비장',
    'POASN': '정상 복수',
    'POREN': '정상 생식소',
    'POKIN': '정상 신장'
}