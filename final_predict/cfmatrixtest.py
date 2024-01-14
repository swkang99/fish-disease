import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 가상의 데이터 생성
classes = ['cat', 'dog', 'bird']
y_true = np.array(['cat', 'dog', 'cat', 'dog', 'bird', 'bird'])
y_pred = np.array(['dog', 'dog', 'cat', 'dog', 'bird', 'cat'])

# 컨퓨전 매트릭스 계산
cm = confusion_matrix(y_true, y_pred, labels=classes)

# 행렬을 전치하여 x축과 y축을 교체
cm_transposed = cm.T

# Seaborn을 사용하여 히트맵 그리기
plt.figure(figsize=(8, 6))
sns.heatmap(cm_transposed, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Confusion Matrix')

# 이미지 파일로 저장
plt.savefig('confusion_matrix_transposed.png')

# 화면에 히트맵 표시
plt.show()
