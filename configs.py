import os

device = "cuda"

img_dir = "./images"
result_img_dir = "./results/model"
raw_img_dir = "./results/raw"
cropped_img_dir = "./results/crop"

alphabet = ' "%&\'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyzÀÁÂÃÉÊÌÍÒÓÔÙÚÝàáâãèéêìíòóôõùúýĂăĐđĩŨũƠơƯưẠạẢảẤấẦầẨẩẫẬậẮắẰằẳẶặẹẺẻẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỶỷỸỹ'
node_labels = ['OTHER', 'ADDRESS', 'SELLER', 'TIMESTAMP', 'TOTAL_COST']

text_detection_weights_path = "./weights/text_detect/craft_mlt_25k_1.pth"
saliency_weight_path = "./weights/saliency/u2netp.pth"
kie_weight_path = "./weights/kie/kie_mcocr.pkl"

saliency_ths = 0.5
score_ths = 0.82
get_max = True  # get max score / filter predicted categories by score threshold
merge_text = True  # if True, concatenate text contents from left to right
infer_batch_vietocr = True  # inference with batch
visualize = False

labels=['e-price',
 'e-unit',
 'e-seller',
 'b-unit',
 'b-unit price',
 'i-price',
 'e-item',
 'i-total_cost',
 'i-time',
 'b-item',
 'i-phone',
 'b-seller',
 'i-item',
 'b-time',
 'i-unit',
 'i-address',
 's-seller',
 'b-phone',
 's-unit',
 's-unit price',
 's-time',
 'b-unit price',
 'i-seller',
 'b-total_cost',
 'i-unit price',
 's-total_cost',
 'b-price',
 'other',
 'e-total_cost',
 'e-phone',
 's-price',
 'e-time',
 'b-address',
 'e-address',
 's-phone',
 's-item']
