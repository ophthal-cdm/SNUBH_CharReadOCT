import argparse, cv2, csv, os
from PIL import Image
import numpy as np
import tensorflow as tf
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from net_utils import CTCLabelConverter, AttnLabelConverter
from net_dataset import RawDataset, AlignCollate
from net_model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


Thick_Right_value = np.arange(32,54)
Thick_Right_value[5] = 41
Thick_Right_value[6] = 42
Thick_Right_value[9] = 37
Thick_Right_value[10] = 38
Thick_Right_value[13] = 49
Thick_Right_value[14] = 50
Thick_Right_value[17] = 45
Thick_Right_value[18] = 46
tmp_pic_idx = 0

class str_recon(object):
    def __init__(self):
        self.image_folder = 'demo3'
        self.workers = 0
        self.batch_size = 32
        self.saved_model = './str3_recognition.pth'
        """ Data processing """
        self.batch_max_length = 50
        self.imgH = 32
        self.imgW = 1500
        self.rgb = False
        self.character = "0123456789abcdefghijklmnopqrstuvwxyz\"#$%&'()*+,-./:;?@[\\]^_`{|}~"
        self.sensitive = False
        self.PAD = False
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling =  'BiLSTM'
        self.Prediction = 'Attn'
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 1024
        self.hidden_size = 512
        self.num_gpu = 1

str_recogni_opt = str_recon()
cudnn.benchmark = True
cudnn.deterministic = True


#model load input : args output : model, data converter
def model_load(opt):
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)


    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    return model, converter

#separate text region with each line input : numpy array, output : numpy array imgs
def str_separated_line(str_region):
    #str_region is black background
    zero_str_region = np.zeros([str_region.shape[0]+4, str_region.shape[1]+4])
    zero_str_region[2:-2,2:-2] = str_region
    # str_region = zero_str_region[2:-2,2:-2]

    str_align_a = (zero_str_region.sum(axis=1) > 0) * 1
    str_align_b = np.zeros(str_align_a.shape)
    str_align_b[1:] = str_align_a[:-1]
    str_result = np.nonzero(str_align_b - str_align_a)
    return [zero_str_region[:str_result[0][1] + 1, :], zero_str_region[str_result[0][2]-1:, :]]

#align text region for model input input : numpy array, output : numpy array
def str_align(str_region):
    #input is white digit black background
    aligned_split_regions = []
    nonzero_region = np.nonzero(str_region)
    # if (nonzero_region[0].size == 0 or nonzero_region[1].size == 0):
    #     return None
    T,L = [nonzero_region[0].min(), nonzero_region[1].min()]
    B,R = [nonzero_region[0].max()+1, nonzero_region[1].max()+1]

    tmp_region = np.zeros([B - T + 10, 15 + R - L])
    tmp_region[5:-5, 5: -10] = str_region[T:B, L:R]
    tmp_region[7:-7, -3:-1] = 255

    # tmp_region = np.zeros([B - T + 10, 15 + R - L])
    # tmp_region[5:-5, 5: -10] = str_region[T:B, L:R]
    # tmp_region[7:-7, -3:-1] = 255
    default_tmp = 0

    # aligned_split_regions.append(Image.fromarray(tmp_region))
    return Image.fromarray(tmp_region)



    # return aligned_split_regions

#inference data input : args, torch model, converter, numpy array imgs output : pred_result float tensor, result str list, max float in tensor, aligned imgs numpy array
def torchmodel(opt, model, converter, img):
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    # str_align_time1 = datetime.now()
    if isinstance(img, list) == True:
        img_list = [str_align(i) for i in img]
    else:
        img_list = [str_align(img)]
    # if None in img_list:
    # AlignCollate_demo_time = datetime.now()
    # print('str_align_time : %s'% (AlignCollate_demo_time - str_align_time1).__str__())
    if img_list.__len__() == 0:
        return img_list
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=True)
    # RawDataset_gen_time = datetime.now()
    # print('AlignCollate_demo_time : %s' % (RawDataset_gen_time - AlignCollate_demo_time).__str__())
    demo_data = RawDataset(image_list=img_list, opt=opt)  # use RawDataset
    # DataLoader_gen_time = datetime.now()
    # print('RawDataset_gen_time : %s' % (DataLoader_gen_time - RawDataset_gen_time).__str__())
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)
    # Pytorch_time_before = datetime.now()
    # print('DataLoader_gen_time : %s' % (Pytorch_time - DataLoader_gen_time).__str__())
    # predict
    pred_result = ''
    result_list = ["" for i in range(img_list.__len__())]
    for image_tensors, image_path_list in demo_loader:
        # no_grad_time = datetime.now()
        # print('no_grad_time : %s' % (no_grad_time - Pytorch_time_before).__str__())
        batch_size = image_tensors.size(0)
        batch_size_time = datetime.now()
        # print('batch_size_time : %s' % (batch_size_time - no_grad_time).__str__())
        image = image_tensors.to(device)
        # image_to_tensor_gen_time = datetime.now()
        # print('image_to_tensor_gen_time : %s' % (image_to_tensor_gen_time - batch_size_time).__str__())
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        # forwawrd_before_time = datetime.now()
        # print('tensor_gen_time : %s' % (forwawrd_before_time - image_to_tensor_gen_time).__str__())
        preds = model(image, text_for_pred, is_train=False)
        # forwawrd_time = datetime.now()
        # print('forwawrd_time : %s' % (forwawrd_time - forwawrd_before_time).__str__())

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        # decode_time = datetime.now()
        # print('decode_time : %s' % (decode_time - forwawrd_time).__str__())



        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        image_idx = 0
        for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
            # cv2.imwrite(str(image_idx) + ".png", image[image_idx].permute(1, cur_idx2, 0).cpu().data.numpy() * 255)


            if 'Attn' in opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                # pred_max_prob = pred_max_prob[:pred_EOS]
                pred_result += str(pred)

                result_list[image_idx] = str(pred)
            image_idx += 1
                # print pred, pred_max_prob
                # print pred_max_prob
            #
            # # calculate confidence score (= multiply of pred_max_prob)
            # confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            #
            # print('{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
            # log.write('{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')
    # Pytorch_time = datetime.now()
    # print('Pytorch_time : %s' % (Pytorch_time - Pytorch_time_before).__str__())
    return pred_result, result_list, preds_max_prob, img_list

#not use
# def split_text(data, canvas, region_path, report_type = "Bi"):
#     par_count = -1
#     tmp_word = ''
#     min_value = 0
#     text_split = []
#     for data_idx, canvas_ele in enumerate(data['line_num']):
#
#         if canvas_ele == par_count:
#             if int(data['conf'][data_idx]) > -1:
#                 min_value = canvas[data['top'][data_idx]:data['top'][data_idx] + data['height'][data_idx],
#                             -50:-40].min()
#
#
#                 if tmp_word:
#                     tmp_word += unicode(' ' + data['text'][data_idx])
#                 else:
#                     tmp_word = unicode(data['text'][data_idx])
#         else:
#             par_count = canvas_ele
#
#             if tmp_word:
#
#                 # if roi_path[min_value][1].find("Date") > -1:
#                 # tmp_word, text_element = strtoBOD(tmp_word, text_element)
#                 text_split.append([region_path[min_value][0], region_path[min_value][2], tmp_word])
#                 tmp_word = ''
#
#             if int(data['conf'][data_idx]) > -1:
#                 # tmp_word = unicode(data['text'][data_idx].encode('UTF-8')
#                 tmp_word = unicode(data['text'][data_idx])
#     if tmp_word:
#         text_split.append([region_path[min_value][0], region_path[min_value][2], tmp_word])
#     return text_split
#
# def tmp_create_csv(ocr_res, xlsx, none_idx, csv, fir_text = 'Filename', sec_text = 'Manufacturer'):
#     tmp_line = []
#     tmp_line.append(fir_text)
#     tmp_line.append(sec_text)
#     if fir_text != 'Filename':
#         for k in ocr_res:
#             tmp_line.append(k[2])
#         for k in none_idx:
#             tmp_line.insert(k, 'None')
#     else:
#         for k in ocr_res:
#             tmp_line.append(k[1])
#     csv.writerow(tmp_line)
#     for idx, k in enumerate(ocr_res):
#         xlsx.write('B%d' % (idx + 2), k[2])
#     return tmp_line

# sheet_count = [1 for i in range(11)]
# crop_name = [[] for i in range(10)]
#separte using connected component and inference tensorflow model input : numpy array, tensorflow model
def crop_to_text(img, model, LTRB_pos = None):

    result = ''
    if LTRB_pos == None:
        LTRB_pos = cv2.connectedComponentsWithStats(img.astype(np.uint8))[2][1:]
    sort_list = np.argsort(LTRB_pos[:,0])
    y_base_shape = float(LTRB_pos[:, 3].max())
    x_base_shape = float(LTRB_pos[:, 2].max())
    for sort_pos in sort_list:
        m_sort = LTRB_pos[sort_pos]
        label_img = img[m_sort[1]:m_sort[1] + m_sort[3], m_sort[0]:m_sort[0] + m_sort[2]]
        tmp_label = np.zeros([28, 28])
        x_len = float(label_img.shape[1])
        y_len = float(label_img.shape[0])
        # if(x_len == 0 or y_len <3):
        #     return 'not'
        if label_img.shape[0] > 18 or label_img.shape[1] > 18:
            if label_img.shape[0] / label_img.shape[1] > 0:
                label_img = cv2.resize(label_img.astype('uint8'),
                                       (18 * label_img.shape[1] / label_img.shape[0], 18))
            else:
                label_img = cv2.resize(label_img.astype('uint8'),
                                       (18, 18 * label_img.shape[0] / label_img.shape[1]))
        else:
            label_img = label_img.astype('uint8')
        if y_len / x_len >= 1:
            label_img = cv2.resize(label_img.astype('uint8'),
                                   (int((18 * (y_len / y_base_shape)) * (x_len/y_len)) , int(18 * (y_len / y_base_shape))),interpolation=cv2.INTER_NEAREST)
        else:
            label_img = cv2.resize(label_img.astype('uint8'),
                                   (int(18 *(x_len / x_base_shape)),int((18 * (x_len / x_base_shape)) * (y_len/x_len))), interpolation=cv2.INTER_NEAREST)
        label_img = cv2.threshold(label_img, 125, 255, cv2.THRESH_BINARY)[1]
        tmp_label[14 - int(round(label_img.shape[0] / 2.)):14 + label_img.shape[0] / 2,
        14 - int(round(label_img.shape[1] / 2.)):14 + label_img.shape[1] / 2] = label_img

        #if using claissfication dot and dash(., -)
        # if tmp_result <= 9:
        # if tmp_result == 10:
        #
        #     result += str(".")
        # elif tmp_result == 11:
        #     None_count += 1
        #     result += str("-")

        if label_img.shape[1]/label_img.shape[0] < 0.8:
            # cur_time = datetime.now()
            tmp_result = model.predict(tmp_label.reshape([1, 28, 28, 1]) / 255.).argmax()
            # print datetime.now() - cur_time
            tf.keras.backend.clear_session()
            result += str(tmp_result)

            ####make annotaor set
            # sheet_count[int(tmp_result)] += 1
            #
            # cv2.imwrite('./tmp_test_20200107/%06d.png' % (sheet_count[-1]), tmp_label)
            # crop_name[int(tmp_result)].append('./tmp_test_20200107/%06d.png' % (sheet_count[-1]))
            # if(sheet_count[int(tmp_result)] > 2800):
            #     continue
            # y_scale = cell_height / float(tmp_label.shape[0])
            # x_scale = cell_width / float(tmp_label.shape[1])
            # worksheet_eval[int(tmp_result)].insert_image('A%d' % (sheet_count[int(tmp_result)]),
            #                                              './tmp_test_20200107/%06d.png' % (sheet_count[-1]),
            #                                              {'x_scale': x_scale, 'y_scale': y_scale})
            # worksheet_eval[int(tmp_result)].write('B%d' % (sheet_count[int(tmp_result)]), tmp_result)
            # sheet_count[-1] += 1
            ####make annotaor
    if(result.__len__() > 1):
        if result[0] == '0' and result[1].isdigit() == True:
            return 'None'
        elif result == '...':
            return 'None'
        tmp_label = []
        label_img = []
        #not using maybe
        # if label_img.sum()/255 > 6:
        #     tmp_label = np.zeros([28, 28])
        #     if label_img.shape[0] / label_img.shape[1] > 0:
        #         label_img = cv2.resize(label_img.astype('uint8'),
        #                                (18 * label_img.shape[1] / label_img.shape[0], 18))
        #     else:
        #         label_img = cv2.resize(label_img.astype('uint8'),
        #                                (18, 18 * label_img.shape[0] / label_img.shape[1]))
        #     label_img = cv2.threshold(label_img, 125, 255, cv2.THRESH_BINARY)[1]
        #     tmp_label[14 - int(round(label_img.shape[0] / 2.)):14 + label_img.shape[0] / 2,
        #     14 - int(round(label_img.shape[1] / 2.)):14 + label_img.shape[1] / 2] = label_img
        #     result += str(model.predict(tmp_label.reshape([1,28,28,1])).argmax())
        # else:
        #     result += '.'

    return result

#read csv file input : record path
def read_pos_record(record_path):
    record = []
    with open(record_path, 'r') as f:
        csv_writer = csv.reader(f, skipinitialspace = False, delimiter=',', quoting=csv.QUOTE_NONE, quotechar='|')
        for row in csv_writer:
            # print cur_line
            row = [m_ele for m_ele in row if m_ele]
            if row.__len__() < 1:
                break
            TL = [int(row[3]), int(row[4])]
            BR = [int(row[5]), int(row[6])]
            if row[0] == 'None':
                record.append([row[0], row[1], row[2], TL, BR])
            else:
                record.append([int(row[0]), row[1], row[2], TL, BR])
    return record

#find report type from title string input : string
def connection_report_path(summary_string):
    if summary_string == 'change':
        return 'THICKNESS MAP CHANGE REPORT, RECENT FOLLOW-UP/'
    elif summary_string == 'overview':
        return 'OVERVIEW REPORT/'
    elif summary_string == 'RNFL':
        return 'RNFL SINGLE EXAM REPORT OU/'
    elif summary_string == 'single':
        return 'THICKNESS MAP SINGLE EXAM REPORT/'

def write_dictonary_type(roi_record, excel_obj):
   ##excel_obj should be input sheet
    roi_path = read_pos_record(roi_record)
    return roi_path

#find report type from title string input : report type str, img shape numpy array, record path list
def load_roi_path(key_obj, img_shape, roi_info_path, diagnosis = False):
    img_key = str(img_shape[0]) + 'X' + str(img_shape[1])
    report_ans = []
    key_split = key_obj.split("_")
    return_roi_key = ""
    for roi_info_path_idx in roi_info_path:
        flags = False
        roi_info_path_split_idx = roi_info_path_idx.split("/")[-1]
        for key_split_idx in key_split:
            if key_split_idx in roi_info_path_split_idx.upper() and img_key in roi_info_path_split_idx.upper() and ("DIAGNOSIS" in roi_info_path_split_idx.upper()) == diagnosis:
                flags = True
            else:
                flags = False
                break
        if flags == True:
            report_ans.append(roi_info_path_idx)
    if report_ans.__len__() > 0:
        return report_ans[0]
    else:
        return None

#split report
def split_report_path(report_path):
    image_path = report_path.split("/")[-1]
    if image_path.find("_") > -1:
        study_date_split = image_path.split("_")[0][:8]
        study_date = "{}-{}-{}".format(study_date_split[:4], study_date_split[4:6], study_date_split[6:8])
        stduy_fn = image_path.split("_")[1] + '_' + image_path.split("_")[0] + image_path[-5]
    else:
        study_date_split = image_path[8:16]
        study_date = "{}-{}-{}".format(study_date_split[:4], study_date_split[4:6], study_date_split[6:8])
        stduy_fn = image_path[:8] + '_' + image_path[8:] + image_path[-5]
    return study_date, stduy_fn

#process thickness report or OVERVIEW input : str model, img, info list numpy array, xlsx obj, row position int, report type
def thickness_process(model, img_list, xlsx_sheet, sheet_row, roi_key):
    global tmp_pic_idx
    try:
        _, result, _2, img_trans = torchmodel(model[1], model[0], model[2], img_list[:, 3].tolist())
    except:
        return None
    c = datetime.now()
    img_list[:, 3] = img_trans
    img_list[:, 4] = result
    cur_idx = 0

    if roi_key.find("THICK") > -1:
        Laterality_idx = img_list[:,2].tolist().index("Laterality")
        if result[Laterality_idx].find("od") > -1:
            img_list[Laterality_idx + 1:Laterality_idx + 1 + Thick_Right_value.shape[0], 0] = Thick_Right_value
            img_list[:,2] = np.core.defchararray.replace(img_list[:,2].astype(np.str), 'Left', 'Right')

    for cur_image_list_index in img_list:
        if cur_image_list_index[2].find("center") > -1:
            cur_image_list_index[4] = cur_image_list_index[4][:-2]
        xlsx_sheet.write(sheet_row + 3, cur_image_list_index[0] - 1, cur_image_list_index[4])
        tmp_pic_idx += 1
        cur_idx += 1

    # for cur_image_list_index in img_list:
    #     if cur_image_list_index[2].find("center") > -1:
    #         cur_image_list_index[4] = cur_image_list_index[4][:-2]
    #     # xlsx_sheet.write(sheet_row + 3, cur_image_list_index[0] - 1, cur_image_list_index[4])
    #
    #     cur_eval_image = cur_image_list_index[3].resize((72, 24), Image.BICUBIC).convert("RGB")
    #     cur_eval_image = np.array(cur_eval_image)
    #     cv2.imwrite("./eval_tmp/" + str(tmp_pic_idx) + "_eval_img_tmp.png", cur_eval_image)
    #     xlsx_sheet.insert_image(cur_idx, 0, "./eval_tmp/" + str(tmp_pic_idx) + "_eval_img_tmp.png",
    #                                 {'x_scale': 1, 'y_scale': 1})
    #     tmp_pic_idx += 1
    #     xlsx_sheet.write(cur_idx, 1, cur_image_list_index[4])
    #     cur_idx += 1
    return sheet_row + 1

#process rnfl report input : str model, num model, img, info list numpy array, xlsx obj, row position int
def RNFL_process(str_model ,model, img_list, xlsx_sheet, sheet_row):
    global tmp_pic_idx
    cur_idx = 0
    value_list = []
    str_list = []
    for img_list_idx in img_list:
        if img_list_idx[1] == 'value':
            value_list.append(img_list_idx)
        else:
            str_list.append(img_list_idx)
    str_list = np.array(str_list)
    try:
        _, result, _2, img_trans = torchmodel(str_model[1], str_model[0], str_model[2], str_list[:, 3].tolist())
    except:

        return None
    str_list[:, 3] = img_trans
    str_list[:, 4] = result


    for cur_image_list_index in value_list:
        if cur_image_list_index[1] != "value":
            continue
        tmp_crop = str_separated_line(cur_image_list_index[3])[1]
        result = crop_to_text(tmp_crop, model)
        xlsx_sheet.write(sheet_row + 3, cur_image_list_index[0] - 1, result)
        tmp_pic_idx += 1
        cur_idx += 1

    for cur_image_list_index in str_list:
        xlsx_sheet.write(sheet_row + 3, cur_image_list_index[0] - 1, cur_image_list_index[4])
        tmp_pic_idx += 1
        cur_idx += 1

    # for cur_image_list_index in str_list:
    #     xlsx_sheet.write(sheet_row + 3, cur_image_list_index[0] - 1, cur_image_list_index[4])
    #
    #     cur_eval_image = cur_image_list_index[3].resize((72, 24), Image.BICUBIC).convert("RGB")
    #     cur_eval_w, cur_eval_h = cur_eval_image.size
    #     cur_eval_image = np.array(cur_eval_image)
    #     cv2.imwrite("./eval_tmp/" + str(tmp_pic_idx) + "_eval_img_tmp.png", cur_eval_image)
    #     xlsx_sheet.insert_image(cur_idx, 0, "./eval_tmp/" + str(tmp_pic_idx) + "_eval_img_tmp.png",
    #                             {'x_scale': 1, 'y_scale': 1})
    #     tmp_pic_idx += 1
    #     xlsx_sheet.write(cur_idx, 1, cur_image_list_index[4])
    #     cur_idx += 1


    return sheet_row + 1

#make training set for single digit classification
                # base_shape = c[1:, 3].max()
                # for label_idx, label in enumerate(c[1:]):
                #     # if base_shape*0.7 > label[3]:
                #     #     continue
                #     tmp_label = np.zeros([28,28])
                #     label_img = mask_img[label[1]:label[1]+label[3],label[0]:label[0]+label[2]]
                #
                #     # if label_img.shape[0]/label_img.shape[1] > 0:
                #     #     label_img = cv2.resize(label_img.astype('uint8'),
                #     #                            (18 * label_img.shape[1] / label_img.shape[0], 18))
                #     # else:
                #     #     label_img = cv2.resize(label_img.astype('uint8'),
                #     #                            (18, 18 * label_img.shape[0]/label_img.shape[1]))
                #     if label_img.shape[0] > 18 or label_img.shape[1] > 18:
                #         if label_img.shape[0] / label_img.shape[1] > 0:
                #             label_img = cv2.resize(label_img.astype('uint8'),
                #                                    (18 * label_img.shape[1] / label_img.shape[0], 18))
                #         else:
                #             label_img = cv2.resize(label_img.astype('uint8'),
                #                                (18, 18 * label_img.shape[0]/label_img.shape[1]))
                #     else:
                #         label_img = label_img.astype('uint8')
                #
                #     label_img = cv2.threshold(label_img, 125, 255, cv2.THRESH_BINARY)[1]
                #     tr_tmp += 1
                #     tmp_label[14-int(round(label_img.shape[0]/2.)):14+label_img.shape[0]/2,14-int(round(label_img.shape[1]/2.)):14+label_img.shape[1]/2] = label_img
                #     cv2.imwrite('./tmp_tr/%06d.png' % (tr_tmp), tmp_label)
                #
                #     y_scale = cell_height / float(tmp_label.shape[0])
                #     x_scale = cell_width / float(tmp_label.shape[1])
                #     worksheet_tr.insert_image('A%d' % (tr_tmp), './tmp_tr/%06d.png' % (tr_tmp),
                #                            {'x_scale': x_scale, 'y_scale': y_scale})
                #     result = model.predict((tmp_label/255.).reshape(1,28,28,1)).argmax().astype('uint8')
                #     # if result < 9:
                #
                #     worksheet_tr.write('C%d' % (tr_tmp), result)
                #
                #     worksheet_tr.write('B%d' % (tr_tmp),'./CRAFT/CRAFT-pytorch/tmp_tr/%06d.png' % (tr_tmp))


# tr_tmp += 1
# cv2.imwrite('./tmp_img_add/%06d.png' % (tr_tmp), mask_img)
# y_scale = cell_height / float(mask_img.shape[0])
# x_scale = cell_width / float(mask_img.shape[1])
# worksheet_test.insert_image('A%d' % (tr_tmp), './tmp_img_add/%06d.png' % (tr_tmp),
#                           {'x_scale': x_scale, 'y_scale': y_scale})
# worksheet_test.write('B%d' % (tr_tmp), tmp_mnist_res)


# worksheet_all.write(shell_name + '%d' % (1), roi_idx[1])

# check croppedimage
#     # workbook = xlsxwriter.Workbook(xlsx_fn)  # make xlsx file
#     # worksheet = workbook.add_worksheet()  # make sheet in
#     # worksheet.write('A%d' % (1), 'Cropped_Image')
#     # worksheet.write('B%d' % (1), 'Predicted')
#     # worksheet.write('C%d' % (1), 'T/F')
# for i in range(10):
#     for j, img_name in enumerate(random.sample(crop_name[i], 2500)):
#         # cur_img_name = random.sample(crop_name[i], 2800)
#         cur_img = cv2.imread(img_name)
#         y_scale = cell_height / float(cur_img.shape[0])
#         x_scale = cell_width / float(cur_img.shape[1])
#         worksheet_eval[i].insert_image('A%d' % (j + 1), img_name, {'x_scale': x_scale, 'y_scale': y_scale})
#         worksheet_eval[i].write('B%d' % (j + 1), i)
#         worksheet_eval[i].write('D%d' % (j + 1), img_name)