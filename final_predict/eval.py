import json
import os

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from predict import final_predict

match_label_symptom = {
    '정상' : 'NO',
    '에드워드': 'EP',
    '비브리오': 'VI',
    '연쇄구균': 'SP',
    '활주세균': 'TM',
    # '여윔증': 'EL',
    '스쿠티카': 'MA',
    '바이러스성출혈성패혈증': 'VH'
}

def read_disease_data():
    disease_json_dir = "/media/lifeisbug/hdd/fish/fish_data/flexink_8+9/disease/"
    disease_dict = dict()
    disease_count_dict = dict()
    for disease in os.listdir(disease_json_dir):
        try:
            with open(disease_json_dir + disease, 'r') as file:
                data = json.load(file)
        except:
            print(disease)
            
        # if data['disease_yn'] == 'Y':
        #     try:
        #         print(data['disease_id'])
        #         print(data['disease_name'])
        #     except:
        #         print(disease)
        disease_name = data['disease_name']
        disease_dict[data['disease_id']] = disease_name

    disease_name = data['disease_name']
    if disease_name not in disease_count_dict.keys():
        disease_count_dict[disease_name]  = 1
    else:
        disease_count_dict[disease_name]  += 1 

    for cnt in disease_count_dict.keys():
        print(f"{cnt} : {disease_count_dict[cnt]}")

    return disease_dict


# def count_fish_by_disease():


def eval():
    fish_img_path="/media/lifeisbug/hdd/fish/fish_data/flexink_8+9/images"
    fishes = set()
    for fish_img in os.listdir(fish_img_path):
        fishes.add(fish_img.split('_')[0])

    top1_acc = 0
    top2_acc = 0
    top3_acc = 0
    
    # classes = ['NO','EP','VI','SP','TM','EL','MA', 'VH']
    classes = ['NO','EP','VI','SP','TM','MA', 'VH']
    pred_list = []
    label_list = []

    disease_dict = read_disease_data()

    cnt = 0

    df = pd.DataFrame(columns=['개체명','예측질병','실제질병','TF','예측증상개수','예측증상목록'])

    for fish in disease_dict.keys():

        prog = cnt / len(disease_dict) * 100
        print(f"*******************진행률: {prog}% ({cnt} / {len(disease_dict)})*******************")
        cnt += 1

        label = disease_dict[fish]
        symptom,pred = final_predict(os.path.join(fish_img_path, fish))
        

        pred_code_1 = match_label_symptom[pred[0][0]]
        pred_code_2 = match_label_symptom[pred[1][0]] if len(pred) > 1 else None
        pred_code_3 = match_label_symptom[pred[2][0]] if len(pred) > 1 else None
        
        label_list.append(label)
        pred_list.append(pred_code_1)
        print("\n============================== 5. 예측 평가 ====================================")
        target_value = label
        matching_keys = [key for key, value in match_label_symptom.items() if value == target_value]
        print(f"개체명: {fish}")
        print(f"1순위 예측: {pred[0][0]}")
        print(f"실제 정답: {matching_keys}")
        tf = True if pred_code_1 == label else False
        
        df.loc[len(df)]= [fish, pred[0][0], matching_keys, tf, len(symptom), symptom]

        if pred_code_2 is None or pred_code_3 is None:
            if label == pred_code_1:
                top1_acc += 1
        elif label == pred_code_3:
            top3_acc += 1
        elif label == pred_code_2:
            top2_acc += 1
            top3_acc += 1
        elif label == pred_code_1:
            top1_acc += 1
            top2_acc += 1
            top3_acc += 1       

    total = len(fishes)
    top1 = top1_acc / total
    top2 = top2_acc / total
    top3 = top3_acc / total
    print(f'top1 accuracy: {top1}')
    print(f'top2 accuracy: {top2}')
    print(f'top3 accuracy: {top3}')

    # 컨퓨전 매트릭스 계산
    cm = confusion_matrix(label_list, pred_list, labels=classes)
    cm_transposed = cm.T
    # Seaborn을 사용하여 히트맵 그리기
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_transposed, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    # plt.show()

    df.to_csv('predict.csv', index=False)

if __name__ == '__main__':
    eval()