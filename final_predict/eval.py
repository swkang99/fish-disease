import json
import os

from predict import final_predict

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
            
        if data['disease_yn'] == 'Y':
            try:
                print(data['disease_id'])
                print(data['disease_name'])
            except:
                print(disease)
        
        disease_dict[data['disease_id']] = disease_name

    disease_name = data['disease_name']
    if disease_name not in disease_count_dict.keys():
        disease_count_dict[disease_name]  = 1
    else:
        disease_count_dict[disease_name]  += 1 

    for cnt in disease_count_dict.keys():
        print(f"{cnt} : {disease_count_dict[cnt]}")


# def count_fish_by_disease():
    
    



def eval():
    fish_img_path="/media/lifeisbug/hdd/fish/fish_data/flexink_8+9/images"
    fishes = set()
    for fish_img in os.listdir(fish_img_path):
        fishes.add(fish_img.split('_')[0])

    top1_acc = 0
    top2_acc = 0
    top3_acc = 0
    for fish in disease_dict.keys():
        
        label = disease_dict[fish]
        # print(os.path.join(fish_img_path, fish))
        pred = final_predict(os.path.join(fish_img_path, fish))
        print(pred, fish)

        # if data[fish] in pred:
        #     top3_acc += 1
        #     top2_acc += 1
        #     top1_acc += 1
        # if data[fish] == pred[0] or data[fish] == pred[1]:
        #     top2_acc += 1
        #     top1_acc += 1
        # if data[fish] == pred[0]:
        #     top1_acc += 1       

    total = len(fishes)
    top1 = len(top1_acc) / total
    top2 = len(top2_acc) / total
    top3 = len(top3_acc) / total


if __name__ == '__main__':
    eval()