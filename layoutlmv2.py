import os
import time
from io import BytesIO
import copy
import imageio
import numpy as np
import math
from PIL import Image
from PIL import Image , ImageDraw , ImageFont  
import torch.nn as nn
import torch
# from fastapi import FastAPI, File, UploadFile

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Tính diện tích phần giao nhau
    if (xB - xA + 1) <0:
        return 0
    if ((yB - yA + 1))<0:
        return 0
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # Diện tích của predicted và ground-truth bounding box
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Tính IoU = diện tích phần giao nhau chia diện tích phần tổng hợp
    # Diện tích phần hợp = tổng diện tích trừ diện tích phần giao
    iou = interArea / float(boxAArea)
    iou123= interArea / float(boxBArea)
    if iou==0:
        iou=0
    else:
        iou=iou+iou123
    # Trả về giá trị iou
    return iou
def bb_intersection_over_union_check(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Tính diện tích phần giao nhau
    if (xB - xA + 1) <0:
        return 0
    if ((yB - yA + 1))<0:
        return 0
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # Diện tích của predicted và ground-truth bounding box
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Tính IoU = diện tích phần giao nhau chia diện tích phần tổng hợp
    # Diện tích phần hợp = tổng diện tích trừ diện tích phần giao
    iou = interArea / float(boxAArea)
   
    if iou==0:
        iou=0
    else:
        iou=iou
    # Trả về giá trị iou
    return iou
def no_accent_vietnamese(text):
    text1=[]
    for s in text:
        s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
        s = re.sub(r'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', s)
        s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
        s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
        s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
        s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
        s = re.sub(r'[ìíịỉĩ]', 'i', s)
        s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
        s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
        s = re.sub(r'[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U', s)
        s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
        s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
        s = re.sub(r'[Đ]', 'D', s)
        s = re.sub(r'[đ]', 'd', s)
        text1.append(s)
    return text1

from backend.backend_utils import (
    NpEncoder,
    run_ocr,
    make_warp_img,
    resize_and_pad,
    get_group_text_line,
)
from backend.text_detect.config import craft_config
from backend.saliency.infer import run_saliency
import configs as cf
import cv2
import os
import random
device=cf.device
def normalize_bbox(bbox, width, height):
     return [
         int(1000 * (bbox[0] / width)),
         int(1000 * (bbox[1] / height)),
         int(1000 * (bbox[2] / width)),
         int(1000 * (bbox[3] / height)),
     ]
from typing import List,Tuple

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]
from transformers import LayoutLMv2Processor,LayoutLMv2ForTokenClassification
def get_model():
#     /storage/quangtm/layoutlmv2/DataLayoutLm2/Checkpoints
    processor = LayoutLMv2Processor.from_pretrained("/data/quangtm/Graph_learning/model/layoutlmv2-base-uncased", revision="no_ocr")
    model = LayoutLMv2ForTokenClassification.from_pretrained("/storage/quangtm/KIE/Checkpoints_19_12") 
    model.eval()
    model.to(device)
    return model,processor
import time
color=(255,0,0)
regex = r"^\,\d"
regex1=r"([0-9]{1})"
regex2 = r"^\.\d"
import re
j=0
labels=cf.labels
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}
get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
colors = get_colors(len(labels))
label2color = {label: colors[idx] for idx, label in enumerate(labels)}
j123=0
import torch.nn as nn
import torch
import torch
def predict(path_img,model,processor,net,detector,text_detector):
    with torch.no_grad():
        t_start=time.time()
        path_img=path_img
        img=path_img
        cv2.imwrite("1.jpg",path_img)
        # j123=j123+1
        random_id=j123
        img = img
        random_id=j123
        img = img
        img = resize_and_pad(img, size=1024, pad=False)
        #     imageio.imwrite(os.path.join(cf.raw_img_dir, "{}.png".format(random_id)), img)
        # img = resize_and_pad(img, size=img.shape[0], pad=False)
        #     imageio.imwrite(os.path.join(cf.raw_img_dir, "{}.png".format(random_id)), img)

        # SALIENCY PREDICTION
#         mask_img = run_saliency(net, img)
#         img[~mask_img.astype(bool)] = 0.0

        # TRANSFORM AND WRAP IMAGE
        warped_img = img
        # sub_img_fp = os.path.join(cf.cropped_img_dir, "{}.png".format(random_id))
        # print(sub_img_fp)
        #     warped_img=img
        #     imageio.imwrite(sub_img_fp, warped_img)
        cells, heatmap, textboxes = run_ocr(
            text_detector, detector, warped_img, craft_config
        )
        _, lines = get_group_text_line(heatmap, textboxes)
        # print(_, lines )
        for line_id, cell in zip(lines, cells):
            cell["group_id"] = line_id

        # CROP IMG AND UPDATE FINAL CELLS INFO
        sub_h_img, sub_w_img, _ = warped_img.shape
        img_info = dict()
        img_info["h_origin"] = sub_h_img
        img_info["w_origin"] = sub_w_img
        img_info["cells"] = cells
        group_ids = np.array([i for i in lines])
        merged_cells = []
        thickness = 2
        Box_sort_list_new,line_arr=[],[]
        merge_list=[]
        listtext=[]
        list_boxesa=[]
        bonus=1
        for group_id in range(max(group_ids) + 1):
            indexes = np.where(group_ids == group_id)[0]
            if len(indexes) == 0:
                continue
            filter_cells = [cells[index] for index in indexes]
            filter_cells = sorted(filter_cells, key=lambda x: np.mean(x["poly"][::2]))
            so_list=[]
            text1=filter_cells[0]['vietocr_text']
            text1=text1.replace("-","")
            result = re.match(regex, text1)
            if result:
                result1a=True
            else:
                result = re.match(regex1, text1)
                if result:
                    result1a=True
                else:
                    result = re.match(regex2, text1)
                    if result:
                        result1a=True
                    else:
                        result1a=False

            for i in range(1,len(filter_cells),1):
                text=filter_cells[i]['vietocr_text']
                text=text.replace("-","")
        #         print(text)
                result = re.match(regex, text)
                if result:
                    result1=True
                else:
                    result = re.match(regex1, text)
                    if result:
                        result1=True
                    else:
                        result = re.match(regex2, text)
                        if result:
                            result1=True
                        else:
                            result1=False  
                if (result1a==True) and (result1==True):
        #             so_list.append(text)
                    so_list.append(i-1) 
                    so_list.append(i) 
                result1a=result1
            so_list=list(set(so_list))
            check=False
            boroi=False
            string_list=""
            id_text=-1
            for i in range(len(filter_cells)):
                id_text=id_text+1
                x1b,y1b,x2b,y2b,x3b,y3b,x4b,y4b=filter_cells[i-1]['poly']
                textn=filter_cells[i-1]['vietocr_text']
                group=filter_cells[i-1]['group_id']
                x1,y1,x2,y2,x3,y3,x4,y4=filter_cells[i]['poly']
                text=filter_cells[i]['vietocr_text']
                group=filter_cells[i]['group_id']
                if i not in so_list:
                    if check==True :
                        listtext.append(string_list)
                        list_boxesa.append([left_pont,top_point,x3b,y3b])
                        new_poly=[left_pont,top_point,x2b,y2b,x3b,y3b,right_point,bottom_point]
                        tmp_cell = copy.deepcopy(filter_cells[0])
                        tmp_cell["poly"] = new_poly
                        tmp_cell["vietocr_text"] = string_list
                        tmp_cell["group_id"]=filter_cells[0]["group_id"]
                        merged_cells.append(copy.deepcopy(tmp_cell))
                        string_list=""
                    merged_cells.append(filter_cells[i])
                    listtext.append(text)
                    list_boxesa.append([x1,y1,x3,y3])
                    check=False
                else:
                    if (abs(x2b-x1)>min(abs(x2-x1),abs(x2b-x1b)) ) and (check==True):
                        listtext.append(string_list)
                        list_boxesa.append([left_pont,top_point,x3b,y3b])
                        new_poly=[left_pont,top_point,x2b,y2b,x3b,y3b,right_point,bottom_point]
                        tmp_cell = copy.deepcopy(filter_cells[0])
                        tmp_cell["poly"] = new_poly
                        tmp_cell["vietocr_text"] = string_list
                        tmp_cell["group_id"]=filter_cells[0]["group_id"]
                        merged_cells.append(copy.deepcopy(tmp_cell))
                        string_list=""
                        left_pont,top_point,right_point,bottom_point=x1,y1,x4,y4
                    if check==False:
                        left_pont,top_point,right_point,bottom_point=x1,y1,x4,y4
                    if i == len(filter_cells)-1:
                        string_list=string_list+text
                        listtext.append(string_list)
                        list_boxesa.append([left_pont,top_point,x3,y3])
                        new_poly=[left_pont,top_point,x2,y2,x3,y3,right_point,bottom_point]
                        tmp_cell = copy.deepcopy(filter_cells[0])
                        tmp_cell["poly"] = new_poly
                        tmp_cell["group_id"]=filter_cells[0]["group_id"]
                        tmp_cell["vietocr_text"] = string_list
                        merged_cells.append(copy.deepcopy(tmp_cell))

                    string_list=string_list+text
        #             print("string_list_fil",string_list,"i",i,len(filter_cells))
                    check=True   
        bbox=list_boxesa
        text=listtext
        #     text=no_accent_vietnamese(text)
        j=0
        t2=time.time()
        image=warped_img
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image, "RGBA")

        font = ImageFont.load_default()
        width, height = image.size
        words_example=[]
        boxes_example=[]
        merged_cells_new=[]
        for i in range(len(bbox)):
            x1 = bbox[i][0]
            y1 = bbox[i][1]
            x3 = bbox[i][2]
            y3 = bbox[i][3]
            txt=text[i]
            box = [x1, y1, x3, y3]
            box = normalize_bbox(box, width=width, height=height)
            if len(txt) < 1: 
                continue
            if min(box) < 0 or max(box) > 1000: # another bug in which a box had -4
                continue
            if ((box[3] - box[1]) < 0) or ((box[2] - box[0]) < 0): # another bug in which a box difference was -12
                continue
            merged_cells_new.append(merged_cells[i])
            words_example.append(txt)
            boxes_example.append(box)
        merged_cells=merged_cells_new
        encoded_inputs=processor(image, words_example, boxes=boxes_example,return_tensors="pt")
#         if len(encoded_inputs['input_ids'][0]) >512:
#         if len(encoded_inputs['input_ids'][0]) >512:
        if len(encoded_inputs['input_ids'][0]) >512:
            encoded_inputs1=dict()
            a=torch.tensor([101])
            input_ids= torch.cat((a,encoded_inputs['input_ids'][0][511:]), 0)
            input_ids=torch.unsqueeze(input_ids, 0)
            encoded_inputs1['input_ids']=input_ids
            a=torch.tensor([0])
            input_ids= torch.cat((a,encoded_inputs['token_type_ids'][0][511:]), 0)
            input_ids=torch.unsqueeze(input_ids, 0)
            encoded_inputs1['token_type_ids']=input_ids
            a=torch.tensor([1])
            input_ids= torch.cat((a,encoded_inputs['attention_mask'][0][511:]), 0)
            input_ids=torch.unsqueeze(input_ids, 0)
            encoded_inputs1['attention_mask']=input_ids
            a=torch.tensor([0, 0, 0, 0])
            a=torch.unsqueeze(a, 0)
            input_ids= torch.cat((a,encoded_inputs['bbox'][0][511:]),0)
            input_ids=torch.unsqueeze(input_ids, 0)
            encoded_inputs1['bbox']=input_ids
            encoded_inputs1['image']=encoded_inputs['image']

            encoded_inputs2=dict()
            a=torch.tensor([101])
            input_ids= torch.cat((encoded_inputs['input_ids'][0][:511],a), 0)
            input_ids=torch.unsqueeze(input_ids, 0)
            encoded_inputs2['input_ids']=input_ids
            a=torch.tensor([0])
            input_ids= torch.cat((encoded_inputs['token_type_ids'][0][:511],a), 0)
            input_ids=torch.unsqueeze(input_ids, 0)
            encoded_inputs2['token_type_ids']=input_ids
            a=torch.tensor([1])
            input_ids= torch.cat((encoded_inputs['attention_mask'][0][:511],a), 0)
            input_ids=torch.unsqueeze(input_ids, 0)
            encoded_inputs2['attention_mask']=input_ids
            a=torch.tensor( [1000, 1000, 1000, 1000])
            a=torch.unsqueeze(a, 0)
            input_ids= torch.cat((encoded_inputs['bbox'][0][:511],a),0)
            input_ids=torch.unsqueeze(input_ids, 0)
            encoded_inputs2['bbox']=input_ids
            encoded_inputs2['image']=encoded_inputs['image']
            outputs = model(input_ids=encoded_inputs1['input_ids'].to(device), attention_mask=encoded_inputs1['attention_mask'].to(device),
                        token_type_ids=encoded_inputs1['token_type_ids'].to(device), bbox=encoded_inputs1['bbox'].to(device),
                        image=encoded_inputs1['image'].to(device))
            outputs1 = model(input_ids=encoded_inputs2['input_ids'].to(device), attention_mask=encoded_inputs2['attention_mask'].to(device),
                            token_type_ids=encoded_inputs2['token_type_ids'].to(device), bbox=encoded_inputs2['bbox'].to(device),
                            image=encoded_inputs2['image'].to(device))
            a=outputs1['logits']
            b=outputs['logits']
            a1=a[0][:511]
            b1=b[0][1:]
            output= torch.cat((a1,b1),0)
            output=torch.unsqueeze(output, 0)
            outputs=output

        else:
            outputs = model(input_ids=encoded_inputs['input_ids'].to(device), attention_mask=encoded_inputs['attention_mask'].to(device),
                        token_type_ids=encoded_inputs['token_type_ids'].to(device), bbox=encoded_inputs['bbox'].to(device),
                        image=encoded_inputs['image'].to(device))
            outputs=outputs.logits

        predictions = outputs.argmax(-1).squeeze().tolist()
        get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
        colors = get_colors(len(labels))
        label2color = {label: colors[idx] for idx, label in enumerate(labels)}
        predictions = outputs.argmax(-1).squeeze().tolist()
        token_boxes = encoded_inputs['bbox'].squeeze().tolist()
        true_predictions = [id2label[pred] for idx, pred in enumerate(predictions)]
        true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) ]
        m = nn.Softmax(dim=1)
        logits=outputs
        logits_1 =m(logits)
        logits_label = torch.argmax(logits_1, dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]
        logits_confidence = [values[label].item()/sum(values) for values, label in zip(logits_1[0], logits_label)]
        logits_confidence=np.array([i.item() for i in logits_confidence])*100


        image=warped_img
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image, "RGBA")
        for prediction, box,cf in zip(true_predictions, true_boxes,logits_confidence):
        #     if cf >sum(logits_confidence)/len(logits_confidence):
            predicted_label = (prediction).lower()
            draw.rectangle(box, outline=label2color[predicted_label])
            draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color[predicted_label], font=font)

        t3=time.time()
        print("thoi gian layout",t3-t2)

        token=[]
        for i in (encoded_inputs.input_ids.squeeze().tolist()):
            token.append(processor.tokenizer.decode(i))
        st=0
        list_text=[]
        box_list=[]
        predict=[]
        confident=[]
        group_ids=[]
        for i in true_predictions:
            group_ids.append(i)
        group_ids=np.array(group_ids)
        true_boxes123=np.array([int((i[0]+i[1]+i[3]+i[2])*13) for i in true_boxes])
        true_boxes1234=np.diff(true_boxes123)
        idx_bs=np.where(true_boxes1234!=0)[0]+1
        idx_bs = np.append(idx_bs, len(true_boxes1234)+1)
        #gộp các token được tách ra quay lại với nhau 
        for idx_b in idx_bs:
            ed=idx_b
            token_a=""
            label=[]
            conf=[]
            for i in token[st:ed]:
                token_a=token_a+i.replace("##","")
            for i in group_ids[st:ed]:
                label.append(i)
            label1=np.array(label)
            for i in logits_confidence[st:ed]:
                conf.append(i)

            max_idx_c=0
            label_true_choise= list(set( label1))[0]
            max_id=conf.index(max(conf))
            label_true_choise= label[max_id]

            if len(list(set(label1)))!=1:
                for idx_c in list(set( label1 )):
                    count_tp=label1.tolist().count(idx_c)
                    if count_tp>max_idx_c:
                        max_idx_c=count_tp
                        label_true_choise=idx_c
                enity=dict()
                for i in list(set(label1)):
                    wh=np.where(label1==i)[0]
                    enity[i]=len(wh)
                if label_true_choise=="other":
                    max_id=conf.index(max(conf))
                    label_true_choise= label[max_id]
                    if "price" in enity.keys() or "unit" in enity.keys() or "unitprice" in enity.keys():
                        if group_ids[st-1]=="item" or group_ids[st-2]=="item" or group_ids[st-3] =="item":
                            max_id=0
                            for i_1 in enity:
                                if i_1 !="other":
                                    if max_id < enity[i_1]:
                                        label_true_choise=i_1
                                        max_id=enity[i_1]

            confident.append(conf[max_id])
            predict.append(label_true_choise) 
            box_list.append([int(true_boxes[st][0]), int(true_boxes[st][1]),int(true_boxes[st][2]),int(true_boxes[st][3])])
            list_text.append(token_a)
            st=ed



        list_text=list_text[1:-1]
        box_list=box_list[1:-1]
        predict=predict[1:-1]
        confident=confident[1:-1]

        total_item_labels=["total_received","total_qty","total refunds","total_cost"]
        ignore_item=["price","unit","unitprice","item","subitem"]
        ignore_adress= ["time","address" , "phone", "storename", "seller"]


        i_entity_result_min=100000
        i_entity_result_max=0
        list_cell_item_unit=[]
        list_cell_item_price=[]
        list_cell_item_name=[]
        list_cell_item_unitprice=[]
        list_cell_item_subitem=[]
        list_cell_item_codeitem=[]
        list_cell_total=[]
        list_cell_add=[]
        list_cell_total_name=[]
        list_cell_title=[]
        list_cell_table=[]
        for i in range(len(predict)):
            if predict[i] in total_item_labels:
                text1=merged_cells[i]['vietocr_text']
                text1=text1.replace(":","")
                result = re.match(regex, text1)
                if result:
                    result1a=True
                else:
                    result = re.match(regex1, text1)
                    if result:
                        result1a=True
                    else:
                        result = re.match(regex2, text1)
                        if result:
                            result1a=True
                        else:
                            result1a=False
                if result1a:
                    list_cell_total_unit=dict()
                    list_cell_total_unit['poly']=merged_cells[i]
                    list_cell_total_unit['label']=predict[i]
                    list_cell_total_unit['confident']=confident[i]
                    list_cell_total.append(list_cell_total_unit)  
                else:
                    list_cell_total_unit=dict()
                    list_cell_total_unit['poly']=merged_cells[i]
                    list_cell_total_unit['label']=predict[i]
                    list_cell_total_unit['confident']=confident[i]
                    list_cell_total_name.append(list_cell_total_unit)  

            elif predict[i] in ignore_adress:
                list_cell_add_unit=dict()
                list_cell_add_unit['poly']=merged_cells[i]
                list_cell_add_unit['label']=predict[i]
                list_cell_add_unit['confident']=confident[i]
                list_cell_add.append(list_cell_add_unit)
            elif predict[i] == "price":
                list_cell_item_price_unit=dict()
                list_cell_item_price_unit['poly']=merged_cells[i]
                list_cell_item_price_unit['label']=predict[i]
                list_cell_item_price_unit['confident']=confident[i]
                list_cell_item_price.append(list_cell_item_price_unit)
            elif predict[i] == "unit":
                list_cell_item_unit_unit=dict()
                list_cell_item_unit_unit['poly']=merged_cells[i]
                list_cell_item_unit_unit['label']=predict[i]
                list_cell_item_unit_unit['confident']=confident[i]
                list_cell_item_unit.append(list_cell_item_unit_unit)
            elif predict[i] == "unitprice":
                list_cell_item_unitprice_unit=dict()
                list_cell_item_unitprice_unit['poly']=merged_cells[i]
                list_cell_item_unitprice_unit['label']=predict[i]
                list_cell_item_unitprice_unit['confident']=confident[i]
                list_cell_item_unitprice.append(list_cell_item_unitprice_unit)
            elif predict[i] == "item":
                list_cell_item_name_unit=dict()
                list_cell_item_name_unit['poly']=merged_cells[i]
                list_cell_item_name_unit['label']=predict[i]
                list_cell_item_name_unit['confident']=confident[i]
                list_cell_item_name.append(list_cell_item_name_unit)
            elif predict[i] == "subitem":
                list_cell_item_subitem_unit=dict()
                list_cell_item_subitem_unit['poly']=merged_cells[i]
                list_cell_item_subitem_unit['label']=predict[i]
                list_cell_item_subitem_unit['confident']=confident[i]
                list_cell_item_subitem.append(list_cell_item_subitem_unit)
            elif predict[i] == "codeitem":
                list_cell_item_codeitem_unit=dict()
                list_cell_item_codeitem_unit['poly']=merged_cells[i]
                list_cell_item_codeitem_unit['label']=predict[i]
                list_cell_item_codeitem_unit['confident']=confident[i]
                list_cell_item_codeitem.append(list_cell_item_codeitem_unit)
            elif predict[i] == "title":
                list_cell_title_unit=dict()
                list_cell_title_unit['poly']=merged_cells[i]
                list_cell_title_unit['label']=predict[i]
                list_cell_title_unit['confident']=confident[i]
                list_cell_title.append(list_cell_title_unit)
            elif  predict[i] == "table":
                list_cell_table_unit=dict()
                list_cell_table_unit['poly']=merged_cells[i]
                list_cell_table_unit['label']=predict[i]
                list_cell_table_unit['confident']=confident[i]
                list_cell_table.append(list_cell_table_unit)


        predict123=np.array([i['label'] for i in list_cell_add])

        Entiti_ADD=[]
        for i_label in set(predict123):
            indexes = np.where(predict123==i_label)[0]
            indexes123=np.array([i for i in indexes])
            indexes1234=np.diff(indexes123)
            idx_bs=np.where(indexes1234!=1)[0]+1
            idx_bs=np.append(idx_bs,len(indexes1234))
            st=indexes123[0]
            text=""
            box_poly=[]
            for i in idx_bs:
                group=[]
                ed=indexes123[i]
                for i_1 in list_cell_add[st:ed+1]:
                    gr=i_1['poly']['group_id']
                    group.append(gr)
                group=sorted(group, key=lambda x: x)
                group=np.array([i for i in group])
                group123=np.diff(group)
                idx_bs_123=np.where(group123!=0)[0]+1+st
                idx_bs_123 = np.append(idx_bs_123, len(group)+st)
                st1=st


                for i_2 in idx_bs_123:
                    x1,y1,x2,y2,x3,y3,x4,y4=10000,10000,0,10000,0,0,10000,0
                    ed1=i_2
                    for i_3 in list_cell_add[st1:ed1]:
                        if i_3['label']==i_label:
                            text=text+" "+i_3['poly']['vietocr_text']

                            x1=min(x1,i_3['poly']['poly'][0])
                            y1=min(y1,i_3['poly']['poly'][1])
                            x2=max(x2,i_3['poly']['poly'][2])
                            y2=min(y2,i_3['poly']['poly'][3])
                            x3=max(x3,i_3['poly']['poly'][4])
                            y3=max(y3,i_3['poly']['poly'][5])
                            x4=min(x4,i_3['poly']['poly'][6])
                            y4=max(y4,i_3['poly']['poly'][7])
                            box_pol=[x1,y1,x2,y2,x3,y3,x4,y4]
                    text=text+"<br>"
                    st1=ed1
                    box_poly.append(box_pol)
                st=ed

            text=text.split("<br>")
            text1=""
            box_poly1=[]
            for i in range(len(text)):
                if text[i]!="":
                    text1=text1+text[i]+"<br>"
                    box_poly1.append(box_poly[i])
            ADD_enti=dict()

            ADD_enti['label']=i_label
            ADD_enti['text']=text1
            ADD_enti['poly']=box_poly1
            Entiti_ADD.append(ADD_enti)

        text_new=[]
        label_new=[]
        poly_new=[]
        for i in range(len(Entiti_ADD)):
            label=Entiti_ADD[i]['label']
            text=Entiti_ADD[i]['text'].split("<br>")
            poly=Entiti_ADD[i]['poly']
            for j in range(len(poly)):
                poly1=[poly[j][0],poly[j][1],poly[j][4],poly[j][5]]
                for m in range(j+1,len(poly),1):
                    poly2=[poly[m][0],poly[m][1],poly[m][4],poly[m][5]]
                    a=bb_intersection_over_union(poly1,poly2)
                    if a >1:
                        text_new.append(text[j])

        Entiti_ADD_new=[]
        for i in range(len(Entiti_ADD)):
            label=Entiti_ADD[i]['label']
            text=Entiti_ADD[i]['text'].split("<br>")
            poly=Entiti_ADD[i]['poly']
            ADD_enti=dict()
            text1=[]
            box_poly=[]
            box_pol_1=[]
            text1_new=[]
            x1,y1,x2,y2,x3,y3,x4,y4=10000,10000,0,10000,0,0,10000,0
            for i_text in range(len(text)-1):
                if text[i_text] not in text_new :
                    text1.append(text[i_text])
                    box_poly.append(poly[i_text])
        #             x1=min(x1,poly[i_text][0])
        #             y1=min(y1,poly[i_text][1])
        #             x2=max(x2,poly[i_text][2])
        #             y2=min(y2,poly[i_text][3])
        #             x3=max(x3,poly[i_text][4])
        #             y3=max(y3,poly[i_text][5])
        #             x4=min(x4,poly[i_text][6])
        #             y4=max(y4,poly[i_text][7])
        #             box_pol=[x1,y1,x2,y2,x3,y3,x4,y4]

            box_pol_2=[]
            text2_new=[]
#             group_line=box_poly[0][1]
#             height=box_poly[0][7]-box_poly[0][1]
            if len(box_poly)!=0:
                group_line=box_poly[0][1]
                height=box_poly[0][7]-box_poly[0][1]
            for i_1 in range(len(box_poly)):
                x1=box_poly[i_1][1]
        #         print("text1[i_1]",text1[i_1],"x1",x1 , "group_line+height:",group_line+height)
                if x1>group_line-height and x1 < group_line+height:
                    group_line=box_poly[i_1][7]
                    box_pol_2.append(box_poly[i_1])
                    text2_new.append(text1[i_1])
                else:    
                    box_pol_1.append(box_pol_2)
                    text1_new.append(text2_new)
                    box_pol_2=[]
                    text2_new=[]
                    box_pol_2.append(box_poly[i_1])
                    text2_new.append(text1[i_1])
                if i_1==len(box_poly)-1:
                    box_pol_1.append(box_pol_2)
                    text1_new.append(text2_new)
            text1=[]
            box_pol1=[]
            for i in range(len(box_pol_1)):
                text12=""
                x1,y1,x2,y2,x3,y3,x4,y4=10000,10000,0,10000,0,0,10000,0
                for j in range(len(box_pol_1[i])):
                    text12=text12+text1_new[i][j]+"<br>"
                    x1=min(x1,box_pol_1[i][j][0])
                    y1=min(y1,box_pol_1[i][j][1])
                    x2=max(x2,box_pol_1[i][j][2])
                    y2=min(y2,box_pol_1[i][j][3])
                    x3=max(x3,box_pol_1[i][j][4])
                    y3=max(y3,box_pol_1[i][j][5])
                    x4=min(x4,box_pol_1[i][j][6])
                    y4=max(y4,box_pol_1[i][j][7])
                    box_pol=[x1,y1,x2,y2,x3,y3,x4,y4]
                box_pol1.append(box_pol)
                text1.append(text12)
            ADD_enti['label']=label
            ADD_enti['text']=text1
            ADD_enti['poly']=box_pol1
            ADD_enti['poly_unit']=box_pol_1
            Entiti_ADD_new.append(ADD_enti)

        Entiti_ADD=Entiti_ADD_new


        Entiti_TOTAL_name=[]       
        predict123=np.array([i['label'] for i in list_cell_total_name])
        for i_label in set(predict123):
            indexes = np.where(predict123==i_label)[0]
            indexes123=np.array([i for i in indexes])
            indexes1234=np.diff(indexes123)
            idx_bs=np.where(indexes1234!=1)[0]+1
            idx_bs=np.append(idx_bs,len(indexes1234))
            st=indexes123[0]
            text=""
            max_confident=0
            for i in idx_bs:
                group=[]
                ed=indexes123[i]
                x1,y1,x2,y2,x3,y3,x4,y4=10000,10000,0,10000,0,0,10000,0
                for i in list_cell_total_name[st:ed+1]:
                    text=text+" "+i['poly']['vietocr_text']

                    x1=min(x1,i['poly']['poly'][0])
                    y1=min(y1,i['poly']['poly'][1])
                    x2=max(x2,i['poly']['poly'][2])
                    y2=min(y2,i['poly']['poly'][3])
                    x3=max(x3,i['poly']['poly'][4])
                    y3=max(y3,i['poly']['poly'][5])
                    x4=min(x4,i['poly']['poly'][6])
                    y4=max(y4,i['poly']['poly'][7])
                    box_pol=[x1,y1,x2,y2,x3,y3,x4,y4]
                st=ed

            TOTAL_entity=dict()
            TOTAL_entity['poly']=[box_pol]
            TOTAL_entity['label']=i_label
            TOTAL_entity['text']=[text]
            Entiti_TOTAL_name.append(TOTAL_entity)




        Entiti_TOTAL=[]       
        predict123=np.array([i['label'] for i in list_cell_total])
        for i_label in set(predict123):
            indexes = np.where(predict123==i_label)[0]
            indexes123=np.array([i for i in indexes])
            indexes1234=np.diff(indexes123)
            idx_bs=np.where(indexes1234!=1)[0]+1
            idx_bs=np.append(idx_bs,len(indexes1234))
            st=indexes123[0]
            text=""
            max_confident=0
            for i in idx_bs:
                group=[]
                ed=indexes123[i]
                for i in list_cell_total[st:ed+1]:
                    if i['confident']>max_confident:
                        max_confident=i['confident']
                        text=[i['poly']['vietocr_text']]
                        poly=[i['poly']['poly']]
                st=ed
            TOTAL_entity=dict()
            TOTAL_entity['label']=i_label
            TOTAL_entity['text']=text
            TOTAL_entity['poly']=poly
            Entiti_TOTAL.append(TOTAL_entity)

        group=[]
        for i_1 in list_cell_title:
            gr=i_1['poly']['group_id']
            group.append(gr)
        Entiti_title=[]       
        group=sorted(group, key=lambda x: x)
        group=np.array([i for i in group])
        group123=np.diff(group)
        idx_bs_123=np.where(group123!=0)[0]+1
        idx_bs_123 = np.append(idx_bs_123, len(group))
        st=0
        for i in idx_bs_123:
            ed=i
            text=""
            x1,y1,x2,y2,x3,y3,x4,y4=10000,10000,0,10000,0,0,10000,0
            for j in list_cell_title[st:ed]:
                text=text+" "+j['poly']['vietocr_text']
                x1=min(x1,j['poly']['poly'][0])
                y1=min(y1,j['poly']['poly'][1])
                x2=max(x2,j['poly']['poly'][2])
                y2=min(y2,j['poly']['poly'][3])
                x3=max(x3,j['poly']['poly'][4])
                y3=max(y3,j['poly']['poly'][5])
                x4=min(x4,j['poly']['poly'][6])
                y4=max(y4,j['poly']['poly'][7])
                box_pol=[x1,y1,x2,y2,x3,y3,x4,y4]
                label= j['label']
            Title_entity=dict()
            Title_entity['poly']=[box_pol]
            Title_entity['label']='title'
            Title_entity['text']=[text]
            Entiti_title.append(Title_entity)
            st=i

        group=[]
        for i_1 in list_cell_table:
            gr=i_1['poly']['group_id']
            group.append(gr)
        Entiti_table=[]       
        group=sorted(group, key=lambda x: x)
        group=np.array([i for i in group])
        group123=np.diff(group)
        idx_bs_123=np.where(group123!=0)[0]+1
        idx_bs_123 = np.append(idx_bs_123, len(group))
        st=0
        for i in idx_bs_123:
            ed=i
            text=""
            x1,y1,x2,y2,x3,y3,x4,y4=10000,10000,0,10000,0,0,10000,0
            for j in list_cell_table[st:ed]:
                text=text+" "+j['poly']['vietocr_text']
                x1=min(x1,j['poly']['poly'][0])
                y1=min(y1,j['poly']['poly'][1])
                x2=max(x2,j['poly']['poly'][2])
                y2=min(y2,j['poly']['poly'][3])
                x3=max(x3,j['poly']['poly'][4])
                y3=max(y3,j['poly']['poly'][5])
                x4=min(x4,j['poly']['poly'][6])
                y4=max(y4,j['poly']['poly'][7])
                box_pol=[x1,y1,x2,y2,x3,y3,x4,y4]
            Table_entity=dict()
            Table_entity['poly']=[box_pol]
            Table_entity['label']='table'
            Table_entity['text']=[text]
            Entiti_table.append(Table_entity)
            st=i

        gt_predict_item=[]
        predict123=np.array([i['label'] for i in list_cell_item_name])
        for i_label in set(predict123):
            indexes = np.where(predict123==i_label)[0]
            indexes123=np.array([i for i in indexes])
            indexes1234=np.diff(indexes123)
            idx_bs=np.where(indexes1234!=1)[0]+1
            idx_bs=np.append(idx_bs,len(indexes1234))
            st=indexes123[0]

            for i in idx_bs:
                group=[]
                ed=indexes123[i]
                for i_1 in list_cell_item_name[st:ed+1]:
                    gr=i_1['poly']['group_id']
                    group.append(gr)
                group=sorted(group, key=lambda x: x)
                group=np.array([i for i in group])
                group123=np.diff(group)
                idx_bs_123=np.where(group123!=0)[0]+1+st
                idx_bs_123 = np.append(idx_bs_123, len(group)+st)
                st1=st
                for i_2 in idx_bs_123:
                    text=""
                    ed1=i_2
                    for i_3 in list_cell_item_name[st1:ed1]:
                        text=text+" "+i_3['poly']['vietocr_text']
                    gt_predict_box_new=sorted(list_cell_item_name[st1:ed1], key=lambda x: x['poly']['poly'][0])
                    x1=gt_predict_box_new[0]['poly']['poly'][0]
                    y1=gt_predict_box_new[0]['poly']['poly'][1]
                    x2=gt_predict_box_new[-1]['poly']['poly'][2]
                    y2=gt_predict_box_new[-1]['poly']['poly'][3]
                    x3=gt_predict_box_new[-1]['poly']['poly'][4]
                    y3=gt_predict_box_new[-1]['poly']['poly'][5]
                    x4=gt_predict_box_new[0]['poly']['poly'][6]
                    y4=gt_predict_box_new[0]['poly']['poly'][7]
                    box=[x1,y1,x2,y2,x3,y3,x4,y4]
                    entity=dict()
                    entity['text']=text
                    entity['label']= i_3['label'] 
                    entity['box']=box
                    gt_predict_item.append(entity)
                    st1=ed1
                st=ed


        entity_list=[]
        entity_name_item_list=[]
        for i in gt_predict_item:
            entity=dict()
            x1,y1,x2,y2,x3,y3,x4,y4=i['box']
            Ax=x1
            Ay=y1
            Bx=x2
            By=y2
            a,b=Bx-Ax,By-Ay
            if a!=0:
                b=b/a
                a=1
            c=b*-Ax+a*Ay
            entity_name_item_list.append([i,[a,b,c]])
            entity['item']=[i,[a,b,c]]
            entity['unit']=100000123
            entity['codeitem']=100000123
            entity['price']=100000123
            entity['unitprice']=100000123
            entity['subitem']=100000123

            entity['unit1']=100000123
            entity['price1']=100000123
            entity['unitprice1']=100000123
            entity['codeitem_confident']=0
            entity['unit_confident']=0
            entity['price_confident']=0
            entity['unitprice_confident']=0
            entity['subitem_confident']=0
            entity['codeitem_distance']=100000123
            entity['unit_distance']=100000123
            entity['price_distance']=100000123
            entity['unitprice_distance']=100000123
            entity['subitem_distance']=100000123
            entity_list.append(entity)
        #         try:
#         if len(gt_predict_item)!=0:
        if len(gt_predict_item)!=0:

            x1,y1,x2,y2,x3,y3,x4,y4=gt_predict_item[0]['box']
            Ax=x1
            Ay=y1
            Bx=x2
            By=y2
            a,b=Bx-Ax,By-Ay
            if a!=0:
                b=b/a
                a=1
            c=b*-Ax+a*Ay
            dist_origin1=10000
            dist_origin2=10000
            dist_origin3=10000
            dist_item=0
            avg_line=0
            dist_unit=10000

            dist_unit=10000
            dist_item=0
            avg_line=0
            for i in range(0,len(entity_list)-1,1):
                dist_item=(dist_item+entity_list[i+1]['item'][0]['box'][5]-entity_list[i]['item'][0]['box'][5])
                avg_line=dist_item/(i+1)
        #     for i in range(0,len(list_cell_item_unitprice)-1,1):
        #         dist_unit=min(min_item,list_cell_item_unitprice[i+1]['poly']['poly'][5]-list_cell_item_unitprice[i]['poly']['poly'][5])

            width_item=10000
            for i in range(0,len(entity_list),1):
                width_item=min(width_item,entity_list[i]['item'][0]['box'][5]-entity_list[i]['item'][0]['box'][1])

    
            if len(list_cell_item_price)!=0:
                x1,y1,x2,y2,x3,y3,x4,y4=list_cell_item_price[0]['poly']['poly']
                center_x=x1
                center_y=y4
                dist_origin1=((b*center_x+c-a*center_y))/(math.sqrt(a*a+b*b))
            if len(list_cell_item_unitprice)!=0:
                x1,y1,x2,y2,x3,y3,x4,y4=list_cell_item_unitprice[0]['poly']['poly']
                center_x=x1
                center_y=y4
                dist_origin2=((b*center_x+c-a*center_y))/(math.sqrt(a*a+b*b))
            if len(list_cell_item_unit)!=0:
                x1,y1,x2,y2,x3,y3,x4,y4=list_cell_item_unit[0]['poly']['poly']
                center_x=x1
                center_y=y4
                dist_origin3=((b*center_x+c-a*center_y))/(math.sqrt(a*a+b*b))
            dist_origin=min(abs(dist_origin1),abs(dist_origin2)+10,abs(dist_origin3)+5 )

            dist_origin=min(abs(dist_origin1),abs(dist_origin2)+10,abs(dist_origin3)+5 )
            if dist_origin1<0:
                dist_origin=-dist_origin

            if abs(dist_origin2) < (avg_line+width_item) or  abs(dist_origin2) > (avg_line-width_item):
                dist_origin2=dist_origin2
            else:
                dist_origin2=dist_origin

            if abs(dist_origin1) < (avg_line+width_item) or  abs(dist_origin1) > (avg_line-width_item):
                dist_origin1=dist_origin1
            else:
                dist_origin1=dist_origin


            for i in range(len(list_cell_item_price)):
                x1,y1,x2,y2,x3,y3,x4,y4=list_cell_item_price[i]['poly']['poly']
                center_x=x1
                center_y=y4
                min_id=100000
                j_id=-10000
                for j in range(len(entity_list)):
                    item,coor=entity_list[j]['item']
                    a,b,c=coor  
                    dist=((b*center_x+c-a*center_y))/(math.sqrt(a*a+b*b))-dist_origin
                    if dist_origin<0:
                        if dist<abs(dist_origin)/5 and dist>0:
                            dist=dist-abs(dist_origin)/5
                    dist=abs(dist)          
                    if dist>0:

                        if dist<min_id and entity_list[j]['price']==100000123:
                            min_id=dist
                            j_id=j
                            y_new= [center_x,b*center_x+c/a]
                        elif dist < min_id and entity_list[j]['price']!=100000123:
                            if dist - entity_list[j]['price_distance'] <dist_origin:
                                if list_cell_item_price[i]['confident']-entity_list[j]['price_confident'] >10:
                                    min_id=dist
                                    j_id=j
                                    y_new= [center_x,b*center_x+c/a]
                            if dist < entity_list[j]['price_distance']:
                                min_id=dist
                                j_id=j
                                y_new= [center_x,b*center_x+c/a]
                    else:

                        if abs(dist)<abs(min_id) and entity_list[j]['price']==100000123:
                            min_id=dist
                            j_id=j
                            y_new= [center_x,b*center_x+c/a]
                        elif abs(dist) < abs(min_id) and entity_list[j]['price']!=100000123:
                            if abs(dist - entity_list[j]['price_distance']) <abs(dist_origin):
                                if list_cell_item_price[i]['confident']-entity_list[j]['price_confident'] >10:
                                    min_id=dist
                                    j_id=j
                                    y_new= [center_x,b*center_x+c/a]
                            if abs(dist) < abs(entity_list[j]['price_distance']):
                                min_id=dist
                                j_id=j
                                y_new= [center_x,b*center_x+c/a]
                if j_id!=-10000:

                    item,coor=entity_list[j_id]['item']
                    point1=(item['box'][0],item['box'][1])

                    point2=[1,1,item['box'][0],item['box'][1]]
                    draw.line([(y_new[0],y_new[1]),(point2[2],point2[3])],fill="red",width=1)

                    entity_list[j_id]['price']=list_cell_item_price[i]['poly']['vietocr_text']
                    entity_list[j_id]['price_distance']=min_id
                    entity_list[j_id]['price_confident']=list_cell_item_price[i]['confident']


            for i in range(len(list_cell_item_unitprice)):
                x1,y1,x2,y2,x3,y3,x4,y4=list_cell_item_unitprice[i]['poly']['poly']
                center_x=x1
                center_y=y4
                min_id=100000
                j_id=-10000
                for j in range(len(entity_list)):
                    item,coor=entity_list[j]['item']
                    a,b,c=coor   
                    dist=((b*center_x+c-a*center_y))/(math.sqrt(a*a+b*b))-dist_origin2
                    if dist_origin<0:
                        if dist<abs(dist_origin2)/6 and dist>0:
                            dist=dist-abs(dist_origin2)/6
                    dist=abs(dist)   
                    if dist>0:
                        if dist<min_id and entity_list[j]['unitprice']==100000123:
                            min_id=dist
                            j_id=j
                        elif dist < min_id and entity_list[j]['unitprice']!=100000123:
                            if dist - entity_list[j]['price_distance'] <dist_origin2:
                                if list_cell_item_unitprice[i]['confident']-entity_list[j]['unitprice_confident'] >10:
                                    min_id=dist
                                    j_id=j
                            if dist < entity_list[j]['unitprice_distance']:
                                min_id=dist
                                j_id=j
                    else:

                        if abs(dist)<abs(min_id) and entity_list[j]['unitprice']==100000123:
                            min_id=dist
                            j_id=j
                        elif abs(dist) < abs(min_id) and entity_list[j]['unitprice']!=100000123:
                            if abs(dist - entity_list[j]['price_distance'] )<abs(dist_origin2):
                                if list_cell_item_unitprice[i]['confident']-entity_list[j]['unitprice_confident'] >10:
                                    min_id=dist
                                    j_id=j
                            if abs(dist) < abs(entity_list[j]['unitprice_distance']):
                                min_id=dist
                                j_id=j

                if j_id!=-10000:            
                    entity_list[j_id]['unitprice']=list_cell_item_unitprice[i]['poly']['vietocr_text']
                    entity_list[j_id]['unitprice_distance']=min_id
                    entity_list[j_id]['unitprice_confident']=list_cell_item_unitprice[i]['confident']  

            for i in range(len(list_cell_item_subitem)):
                x1,y1,x2,y2,x3,y3,x4,y4=list_cell_item_subitem[i]['poly']['poly']
                center_x=x1
                center_y=y4
                min_id=100000
                j_id=0
                for j in range(len(entity_list)):
                    item,coor=entity_list[j]['item']
                    a,b,c=coor   
                    dist=((b*center_x+c-a*center_y))/(math.sqrt(a*a+b*b))-dist_origin
                    if dist_origin<0:
                        if dist<abs(dist_origin)/5 and dist>0:
                            dist=dist-abs(dist_origin)/5

                    dist=abs(dist)        
                    if dist>0:
                        if dist<min_id:
                            min_id=dist
                            j_id=j
                            if entity_list[j_id]['subitem']==100000123:
                                entity_list[j_id]['subitem']=""
                    else:
                        if abs(dist)<abs(min_id):
                            min_id=dist
                            j_id=j
                            if entity_list[j_id]['subitem']==100000123:
                                entity_list[j_id]['subitem']=""

                entity_list[j_id]['subitem']=str(entity_list[j_id]['subitem'])+list_cell_item_subitem[i]['poly']['vietocr_text']
                entity_list[j_id]['subitem_distance']=min_id
                entity_list[j_id]['subitem_confident']=list_cell_item_subitem[i]['confident']

            for i in range(len(list_cell_item_unit)):
                x1,y1,x2,y2,x3,y3,x4,y4=list_cell_item_unit[i]['poly']['poly']
                center_x=x1
                center_y=y4
                min_id=100000
                j_id=-10000
                for j in range(len(entity_list)):
                    item,coor=entity_list[j]['item']

                    a,b,c=coor   
                    dist=((b*center_x+c-a*center_y))/(math.sqrt(a*a+b*b))-dist_origin1
                    if dist_origin1<0:
                        if dist<abs(dist_origin1)/5 and dist>0:
                            dist=dist-abs(dist_origin1)/5
        #             print(list_cell_item_unit[i]['poly']['vietocr_text'])
        #             print(item['text'])
        #             print(dist)
        #             print("-----------")
                    dist=abs(dist) 
                    if dist>0:
                        if dist<min_id and entity_list[j]['unit']==100000123:

                            min_id=dist
                            j_id=j
                            y_new= [center_x,b*center_x+c/a]
                        elif dist < min_id and entity_list[j]['unit']!=100000123:
                            if dist - entity_list[j]['unit_distance'] <dist_origin1:
                                if list_cell_item_unit[i]['confident']-entity_list[j]['subitem_confident'] >10:
                                    min_id=dist
                                    j_id=j
                                    y_new= [center_x,b*center_x+c/a]
                            if dist < entity_list[j]['unitprice_distance']:
                                min_id=dist
                                j_id=j
                                y_new= [center_x,b*center_x+c/a]
                    else:

                        if abs(dist)<abs(min_id) and entity_list[j]['unit']==100000123:
                            min_id=dist
                            j_id=j
                            y_new= [center_x,b*center_x+c/a]
                        elif abs(dist) < abs(min_id) and entity_list[j]['unit']!=100000123:
                            if abs(dist - entity_list[j]['unit_distance']) <abs(dist_origin1):
                                if list_cell_item_unit[i]['confident']-entity_list[j]['subitem_confident'] >10:
                                    min_id=dist
                                    j_id=j
                                    y_new= [center_x,b*center_x+c/a]
                            if abs(dist) < abs(entity_list[j]['unitprice_distance']):
                                min_id=dist
                                j_id=j
                                y_new= [center_x,b*center_x+c/a]
                if j_id!=-10000:
                    entity_list[j_id]['unit']=list_cell_item_unit[i]['poly']['vietocr_text']
                    entity_list[j_id]['unit_distance']=min_id
                    entity_list[j_id]['unit_confident']=list_cell_item_unit[i]['confident']  



            for i in range(len(list_cell_item_codeitem)):
                x1,y1,x2,y2,x3,y3,x4,y4=list_cell_item_codeitem[i]['poly']['poly']
                center_x=x1
                center_y=y4
                min_id=100000
                j_id=-10000
                for j in range(len(entity_list)):
                    item,coor=entity_list[j]['item']
                    a,b,c=coor   
                    dist=((b*center_x+c-a*center_y))/(math.sqrt(a*a+b*b))-dist_origin
                    dist=abs(dist)
                    if dist>0:
                        if dist<min_id and entity_list[j]['codeitem']==100000123:
                            min_id=dist
                            j_id=j
                            y_new= [center_x,b*center_x+c/a]
                        elif dist < min_id and entity_list[j]['codeitem']!=100000123:
                            if dist - entity_list[j]['codeitem_distance'] <dist_origin:
                                if list_cell_item_codeitem[i]['confident']-entity_list[j]['subitem_confident'] >10:
                                    min_id=dist
                                    j_id=j
                                    y_new= [center_x,b*center_x+c/a]
                            if dist < entity_list[j]['codeitem_distance']:
                                min_id=dist
                                j_id=j
                                y_new= [center_x,b*center_x+c/a]
                    else:
                        if abs(dist)<abs(min_id) and entity_list[j]['codeitem']==100000123:
                            min_id=dist
                            j_id=j
                            y_new= [center_x,b*center_x+c/a]
                        elif abs(dist) < abs(min_id) and entity_list[j]['codeitem']!=100000123:
                            if abs(dist - entity_list[j]['codeitem_distance']) <abs(dist_origin):
                                if list_cell_item_codeitem[i]['confident']-entity_list[j]['subitem_confident'] >10:
                                    min_id=dist
                                    j_id=j
                                    y_new= [center_x,b*center_x+c/a]
                            if abs(dist) < abs(entity_list[j]['codeitem_distance']):
                                min_id=dist
                                j_id=j
                                y_new= [center_x,b*center_x+c/a]

                if j_id!=-10000:
                    entity_list[j_id]['codeitem']=list_cell_item_codeitem[i]['poly']['vietocr_text']
                    entity_list[j_id]['codeitem_distance']=min_id
                    entity_list[j_id]['codeitem_confident']=list_cell_item_codeitem[i]['confident']  





#         except:
#             entity_list=[]



#             entity_list=[]




        
#         except:
#             entity_list=[]





        #         except:
        #             entity_list=[]

        chuoitext=""
        for  i in Entiti_ADD:
            if i['label']=="storename":
                chuoitext=chuoitext+"<b>Tên Cửa Hàng: </b>"+str(i['text'])
            elif i['label']=="address":
                chuoitext=chuoitext+"<b>Địa Chỉ : </b>"+str(i['text'])
            elif i['label']=="phone":
                chuoitext=chuoitext+"<b>Số Điện Thoại : </b>"+str(i['text'])
            elif i['label']=="seller":
                chuoitext=chuoitext+"<b>Nhân Viên Bán Hàng : </b>"+str(i['text'])
            elif i['label']=="time":
                chuoitext=chuoitext+"<b>Thời Gian : </b>"+str(i['text'])
        chuoitext=chuoitext+"<p></p>"
        for i in Entiti_TOTAL:
            if i['label']=="total_qty":
                chuoitext=chuoitext+"<b>Tổng Số Lượng Sản Phẩm: </b>"+str(i['text'])+"<br>"
            elif i['label']=="total_received":
                chuoitext=chuoitext+"<b>Số Tiền Khách Đưa: </b>"+str(i['text'])+"<br>"
            elif i['label']=="total refunds":
                chuoitext=chuoitext+"<b>Số Tiền Trả Lại:</b>"+str(i['text'])+"<br>"
            elif i['label']=="total_cost":
                chuoitext=chuoitext+"<b>Tổng Tiền: </b>"+str(i['text'])+"<br>"
        chuoitext=chuoitext+"<table style='color: black;' border='3'>   <tr> <th>Sản phẩm</th><th>Số lượng</th><th>Đơn giá</th> <th>Thành tiền</th></tr>"

        for i in entity_list:
            subitem=i['subitem']
            if subitem==100000123:
                subitem=""
            sanpham=str(i['item'][0]['text']) + "<br>"+str(subitem)
            soluong=i['unit']
            gia1sp=i['unitprice']
            giasp=i['price']
            if i['unit']==100000123  :
                soluong="0"
            if i['unitprice']==100000123 :
                gia1sp="0"
            if i['price']==100000123:
                giasp="0"


            chuoitext=chuoitext+ "<tr><td>"+str(sanpham) +"</td><td>" +soluong+"</td><td>"+  gia1sp +"</td><td>"+ giasp+"</td></tr>"

        chuoitext=chuoitext+"</table>" 
        t_end=time.time()
        chuoitext=chuoitext+"</table>" 
        chuoitext=chuoitext+"<p></p>"+"<br>"+"<b>Thời Gian Thực Hiện : </b>"+str(t_end-t_start)

# from IPython.core.display import display, HTML
# image.show()
# display(HTML(chuoitext))
      

                    
                    
        
      
        return image , chuoitext
