# [빅데이터 기반 양식생산성 향상 기술개발] 수산질병 모니터링 및 질병진단/예측 시스템 개발 : 수산질병 예측 알고리즘 개발 (2023)

## 코드 사용법 (how to use code)

1. directory.py를 연다.

2. 데이터셋의 최상위 폴더 경로의 변수를 본인의 경로로 수정한다.

3. demo.ipynb를 연다.

4. 데이터셋을 내려받고 최초 한 번만 1번째 블록을 실행한다.
   개체별로 구분된 이미지가 모두 한 폴더로 모이게 되는 것을 확인한다.
   
5. 2번째 블록을 실행한다.

6. 본인이 사용할 함수를 execute() 함수로 실행한다.
   execute() 함수는 (실행할 함수, 데이터셋, 구분번호)를 인자로 받는다.

---

## 함수 실행 예시 (example function execution)

### 5월 데이터셋의 16일 데이터에서 11번 이미지의 내부 장기 시각화
```
execute_func(seg.visualize_map, data_0516, '11')
```

### 5월 데이터셋의 16일 데이터에서 분류를 위한 장기 이미지 생성
### 반복문을 통해 모든 내부이미지에 대해 실행
```
for num in inner:
    execute(seg.visualize_map, data_0516, num)
```
    
### 반복문 추가 설명
```
for num in inner: # 내부 이미지 전체에 대해 함수 사용

for num in outer: # 외부 이미지 전체에 대해 함수 사용

for num in all_num: # 전체 이미지에 대해 함수 사용
```

---

## 실행 가능한 함수 목록(지속적으로 추가 예정)

### 세그먼테이션 맵 시각화
```
seg.visualize_map
```
### 바운딩박스 시각화
```
det.visualize_box   
```
### YOLO에서 쓸 검출 데이터셋 생성(수정 예정)
```
det.get_box 
```
### 클래스별 통계 출력
```
ut.count_class
```
### 분류용 장기 이미지 생성
```
cls.get_organ # 해당 함수 사용 시, classification.py에서 원하는 값을 수정 후 실행
```
함수의 자세한 설명 및 매개변수는 함수 주석(docstring) 참고

---

## 파일별 기능 설명

utils.py : 정보 검색, 통계 등의 기능

dicts.py : 클래스 코드(딕셔너리)를 관리, 랜덤 마스크 생성 관련 색깔 코드 및 학습에 사용하는 클래스 포함

directory.py : 설정된 경로로부터 데이터를 불러오기

images.py : 이미지 관련 기능

segmentation.py : 분할 데이터셋 처리, 맵 시각화 기능 포함

classification.py : 분류 데이터셋 처리(생성)

detection.py : 검출 데이터셋 처리
