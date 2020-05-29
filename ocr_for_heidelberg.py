import xlsxwriter
import shutil
import glob
import sys
import psutil
from report_extract_utils import *
import os,argparse
from datetime import datetime

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession




reload(sys)
sys.setdefaultencoding('utf-8')

parser = argparse.ArgumentParser(description='Text Detection')
parser.add_argument('--test_path', default="./test/", type=str, help='folder path to input images')
parser.add_argument('--record_path', default='./mask_point_mapping/', type=str, help='folder path to mask position file')
parser.add_argument('--res_path', default="./res/", type=str, help='folder path to input images')
parser.add_argument('--xlsx_name', default='Heidelberg', type=str, help='folder path to mask position file')
parser.add_argument('--split_save', default= True, type=bool, help='folder path to mask position file')
parser.add_argument('--img_read_interval', default= 1000, type=int, help='folder path to mask position file')

saving_text = []
args = parser.parse_args()

test_folder = args.test_path
#read path in directory
report_path = os.listdir(test_folder)
#read record path in directory
record_path = glob.glob(args.record_path + '*.csv')
#model load
str_model, str_converter = model_load(str_recogni_opt)
str_model.eval()
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
num_model = tf.keras.models.load_model('./digit_classifier_1216.h5', custom_objects={'ReLU':tf.keras.layers.ReLU})
num_model.trainable = False


# make result path
if os.path.isdir(args.res_path) == False:
    os.mkdir(args.res_path)
res_path = args.res_path
if os.path.isdir(res_path) == False:
    os.mkdir(res_path)
Figure_path = res_path + 'Figure/'
if os.path.isdir(Figure_path) == False:
    os.mkdir(Figure_path)
None_config_path = res_path + 'None_Config/'
if os.path.isdir(None_config_path) == False:
    os.mkdir(None_config_path)
Not_read_config = res_path + 'not_read/'
if os.path.isdir(Not_read_config) == False:
    os.mkdir(Not_read_config)

prev_time = datetime.now()
#create xlsx file to save result
today_date = str(prev_time.year) + str(prev_time.month).zfill(2) + str(prev_time.day).zfill(2) + str(prev_time.hour).zfill(2) + str(prev_time.minute).zfill(2)
xlsx_result_fn =  res_path + today_date + "_" + args.xlsx_name + '.xlsx'
workbook_result = xlsxwriter.Workbook(xlsx_result_fn, {'constant_memory': True})  # make xlsx file
workbook_result.use_zip64()
worksheet_result = workbook_result.add_worksheet()  # make sheet in

#write measurement index in xlsx file
ocr_f = open('./config_ocr.txt', 'r')
ocr_f_lines = ocr_f.readlines()
for idx, ocr_f_lines_idx in enumerate(ocr_f_lines):
    worksheet_result.write(2, idx + 1, ocr_f_lines_idx.split("\r")[0])
for idx, ocr_f_lines_idx in enumerate(ocr_f_lines):
    worksheet_result.write(3, idx + 1, idx + 1)

#create xlsx file to save recognized report
# xlsx_distribute_fn =  res_path + today_date + 'Heidelberg_distribute.xlsx'
# workbook_file_distribute = xlsxwriter.Workbook(xlsx_distribute_fn, {'constant_memory': True})  # make xlsx file
# workbook_file_distribute.use_zip64()
# worksheet_file_distribute = workbook_file_distribute.add_worksheet()  # make sheet in


roi_paths = []
saveP_Point = 0
tmp_pic_idx = 0
tr_tmp = 3
title_tmp = 1

with torch.no_grad():
    for i, cur_image_path in enumerate(glob.glob(test_folder + "*")):
        try:
            if cur_image_path.split("/")[-1].find(".") == -1:
                continue

            title_tmp += 1
            OD_value = False

            OD_Region = 'None'
            OS_Region = 'None'
            Laterality = 'B'

            #read report image
            cur_image = cv2.imread(cur_image_path)
            org_image = cur_image.copy()

            #preprocessing report file with binary for thickness and RNFL
            cur_image = np.bitwise_and(org_image[:, :, 0] > 80,
                                       np.bitwise_and(org_image[:, :, 1] > 80, org_image[:, :, 2] > 80)).astype(
                'uint8') * 255
            cur_image2 = np.bitwise_or(org_image[:, :, 0] > 70,
                                      np.bitwise_or(org_image[:, :, 1] > 70, org_image[:, :, 2] > 100)).astype(
                'uint8') * 255
            cur_image = cv2.cvtColor(cv2.cvtColor(cur_image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY)
            cur_image2 = cv2.cvtColor(cv2.cvtColor(cur_image2, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY)

            #crop title region
            img_labeled_title = cv2.connectedComponentsWithStats(255-cur_image[:cur_image.shape[0]/5,:cur_image.shape[1]*2/3])[2][1:]
            img_labeled_title = np.array([img_labeled_title_idx for img_labeled_title_idx in img_labeled_title if img_labeled_title_idx[2]> cur_image.shape[1]/3])
            title_line = img_labeled_title[img_labeled_title[:,1].argmin()][1]
            crop_img = (255 - cur_image[:title_line, :cur_image.shape[1]*2/3]).astype(np.uint8)

            #separted each line in region
            title_region = str_separated_line(crop_img)

            #recognize string title region if fail or value==0 that save in not read directory
            image_report_type, _, conf_score, _ = torchmodel(str_recogni_opt, str_model, str_converter, title_region)
            # try:
            #     image_report_type, _, conf_score, _ = torchmodel(str_recogni_opt, str_model, str_converter, title_region)
            # except:
            #     psutil.virtual_memory()
            #     #not read
            #     shutil.move(cur_image_path, Not_read_config + cur_image_path.split("/")[-1])
            #     continue
            if image_report_type.__len__() == 0:
                #not read
                shutil.move(cur_image_path, Not_read_config + cur_image_path.split("/")[-1])
                continue
            #print recognized report count, report path, type
            print i, cur_image_path, image_report_type

            #make roi key
            roi_key = ' '
            roi_path_fn = None
            #ensure report type in recore file, if fail that save in None config path
            if image_report_type.upper().find('RNFL') > -1:
                roi_key = 'RNFL_'
                if image_report_type.upper().find('SIN') > -1 or image_report_type.upper().find('BASIC') > -1:
                    roi_key = "{}{}".format(roi_key, "SINGLE")
                    cur_image = cur_image2
                else:
                    #none config
                    shutil.move(cur_image_path, None_config_path + cur_image_path.split("/")[-1])
                    continue
            elif image_report_type.upper().find('THICK') > -1 or image_report_type.upper().find('TNICK') > -1:
                roi_key = 'THICKNESS_'
                if image_report_type.upper().find('CHANGE') > -1:
                    roi_key = "{}{}".format(roi_key, "CHANGE")
                elif image_report_type.upper().find('SINGLE') > -1:
                    roi_key = "{}{}".format(roi_key, "SINGLE")
                else:
                    img_save_path = res_path + 'None_Config/'
                    if os.path.isdir(img_save_path) == False:
                        os.mkdir(img_save_path)
                    #none conifg
                    shutil.move(cur_image_path, None_config_path + cur_image_path.split("/")[-1])
                    continue
            elif image_report_type.upper().find('OVER') > -1 or image_report_type.upper().find('OVEN') > -1:
                roi_key = 'OVERVIEW'
            elif image_report_type.upper().find('SPEC') > -1 or image_report_type.upper().find('TRACK') > -1:
                #None config
                shutil.move(cur_image_path, None_config_path + cur_image_path.split("/")[-1])
                continue
            else:
                #not read
                shutil.move(cur_image_path, Not_read_config + cur_image_path.split("/")[-1])
                continue
            # worksheet_file_distribute.write(tr_tmp + 3, 1, cur_image_path.split("/")[-1])
            # worksheet_file_distribute.write(tr_tmp + 3, 2, img_save_path)
            ##### not heidelberg, not recorded config file, not recorded diagnosis

            #load roi record file
            roi_path_fn = load_roi_path(roi_key, cur_image.shape[:2], record_path)

            #preprocess strange file
            if roi_path_fn == None or cur_image[-30:,-30:].min() != 255:
                #None config
                shutil.move(cur_image_path, None_config_path + cur_image_path.split("/")[-1])
                continue


            #read record file
            roi_path = read_pos_record(roi_path_fn)
            figure_path = Figure_path + roi_key + '/'
            if os.path.isdir(figure_path) == False:
                os.mkdir(figure_path)

            #check if record has diagnosis region
            reg1 = roi_path[1]
            diagnosis_region = cur_image[reg1[4][0]:reg1[4][0] + int((reg1[4][0]-reg1[3][0])), reg1[3][1]-50:reg1[3][1]+30]
            diagnosis_check = cv2.connectedComponents(255-diagnosis_region)
            if diagnosis_check[0] < 5 :
                roi_path_fn = load_roi_path(roi_key, cur_image.shape[:2], record_path, True)
                if roi_path_fn == None:

                    shutil.move(cur_image_path, None_config_path + cur_image_path.split("/")[-1])
                    continue
                roi_path = write_dictonary_type(roi_path_fn, worksheet_result)


            #extract study date and patient number in report path
            study_date, study_fn = split_report_path(cur_image_path)
            #create list object, saving col number int, str or num string, measuremnet value name string, cropped img, result blank
            cur_image_list = np.zeros([0,5], dtype=np.object)
            # worksheet_result = workbook_result.add_worksheet(str(i) + '-th report')  # make sheet in
            # worksheet_result.set_default_row(height=18)
            # worksheet_result.set_column('A:A', 9.38)

            # worksheet_result.write(0, 1, 'Prediction')
            # worksheet_result.write(0, 2, 'Annotator1')
            # worksheet_result.write(0, 3, 'Annotator2')

            #index cropping masurement value region
            cur_idx = 0
            for idx, roi_idx in enumerate(roi_path):
                #write patint number, study date, report type
                if 3 in [roi_idx[0]]:
                    worksheet_result.write(tr_tmp + 3, roi_idx[0] - 1, study_fn.split("_")[0])
                    continue
                if 4 in [roi_idx[0]]:
                    worksheet_result.write(tr_tmp + 3, roi_idx[0]- 1, study_date)
                    continue
                if 8 in [roi_idx[0]]:
                    worksheet_result.write(tr_tmp + 3, roi_idx[0]- 1, roi_key)
                    continue

                #crop boundingbox
                tmp_cur_image_list = np.zeros([1, 5], dtype=np.object)
                crop = cur_image[roi_idx[3][0]:roi_idx[4][0],roi_idx[3][1]:roi_idx[4][1]].copy()
                if roi_idx[2].find("figure")>-1 or roi_idx[2].find("Figure")>-1:
                    # continue
                    color_crop = org_image[roi_idx[3][0]:roi_idx[4][0],roi_idx[3][1]:roi_idx[4][1]]
                    figure_fn = figure_path + study_date + '_' + roi_idx[2] + '.png'
                    cv2.imwrite(figure_fn, color_crop)
                    worksheet_result.write(tr_tmp+3, roi_idx[0] - 1, figure_fn)
                    continue

                if "RNFL" in roi_key:
                    #preprocessing for RNFL
                    if roi_idx[2].find("OD_Region") > -1:
                        OD_Region = crop
                        continue
                    if roi_idx[2].find("OS_Region") > -1:
                        OS_Region = crop
                        continue

                    if OD_Region != 'None' and OS_Region != 'None':
                        if OD_Region.min()  ==  0 and OS_Region.min()  ==  0:
                            Laterality = 'B'
                            OPT = 'B'
                        elif OD_Region.min()  ==  0 and OS_Region.min()  !=  0:
                            Laterality = 'Left'
                            OPT = 'OS'
                        elif OD_Region.min()  !=  0 and OS_Region.min()  ==  0:
                            Laterality = 'Right'
                            OPT = 'OD'

                    if Laterality != 'B':
                        if roi_idx[2].find(Laterality) > -1:
                            continue
                        if roi_idx[2].find(OPT) > -1:
                            continue
                tmp_cur_image_list[0][0] = roi_idx[0]
                tmp_cur_image_list[0][1] = roi_idx[1]
                tmp_cur_image_list[0][2] = roi_idx[2]
                tmp_cur_image_list[0][3] = (255 - crop)
                cur_image_list = np.vstack((cur_image_list, tmp_cur_image_list))

            if "RNFL" in roi_key:
                #doing OCR RNFL report, input : [str model], num_model, cropped image list, xlsx obj, writing xlsx row int
                recogni_result = RNFL_process([str_model, str_recogni_opt, str_converter], num_model,  cur_image_list, worksheet_result, tr_tmp)
            else:
                #doing OCR RNFL report, input : [str model], croopped image list, xlsx obj, writing xlsx row int, report type string
                recogni_result = thickness_process([str_model, str_recogni_opt, str_converter], cur_image_list, worksheet_result, tr_tmp, roi_key)
            if recogni_result != None:
                tr_tmp = recogni_result
            else:
                #None config
                shutil.move(cur_image_path, None_config_path + cur_image_path.split("/")[-1])
                continue
        except:
            shutil.move(cur_image_path, None_config_path + cur_image_path.split("/")[-1])



workbook_result.close()
# workbook_file_distribute.close()

