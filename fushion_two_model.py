import numpy as np
import os, copy

test_model = "val"

head = ["image_id", "image_category", "neckline_left", "neckline_right", "center_front",
        "shoulder_left", "shoulder_right", "armpit_left", "armpit_right", "waistline_left", "waistline_right",
        "cuff_left_in", "cuff_left_out", "cuff_right_in", "cuff_right_out", "top_hem_left", "top_hem_right",
        "waistband_left", "waistband_right", "hemline_left", "hemline_right", "crotch", "bottom_left_in",
        "bottom_left_out", "bottom_right_in", "bottom_right_out"]
result_csv = [head]
data_attr = {
    "blouse": [2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16],  # 13 keypoints
    'dress': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20],  # 15 keypoints
    'outwear': [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # 14 keypoints
    'skirt': [17, 18, 19, 20],  # 4 keypoints
    'trousers': [17, 18, 21, 22, 23, 24, 25]  # 7 keypoints
}

annotations_path = {
    #3.7_result_prob
    "val_base_predict1": os.path.join("submit", "val", "3.7_result_prob.csv"),
    "val_base_predict2": os.path.join("submit", "val", "daiwei_val_3.986_prob.csv"),
    "test_base_predict1": os.path.join("submit", "test", "liuzhenhua_result_test_4.036_prob.csv"),
    "test_base_predict2": os.path.join("submit", "test", "daiwei_result_test_3.986_prob.csv"),
    "gt": os.path.join("submit/val/val.csv"),
}

def read_data(filename, flag = False):
    input_file = open(filename, 'r')
    data_dict = {}

    i = 0
    for line in input_file:
        if i == 0:  ### drop  the header
            i = 1
            continue
        line = line.strip()
        line = line.split(',')
        name = line[0]
        name = name.replace("/data/daiwei/FashionAI/test_b/", "")
        name = name.replace("/data/daiwei/FashionAI/test/", "")
        type = line[1]

        def fn(x):
            c = x.split('_')
            if flag:
                return [round(float(r)) for r in c]
            else:
                return [float(r)for r in c]

        joints = list(map(fn, line[2:]))
        joints = np.reshape(joints, (-1, 3))
        data_dict[name] = {'joints': joints, 'type': type}
    input_file.close()
    return data_dict

def calculate_norm(gt_data):
    samples = len(gt_data.keys())
    norm_mat = np.zeros((samples), np.float)
    for i, name in enumerate(gt_data.keys()):
        catgory = gt_data[name]['type']
        pts = gt_data[name]['joints']
        if catgory == 'dress' or catgory == 'outwear' or catgory == 'blouse':
            norm = np.sqrt(np.square(pts[5][0] - pts[6][0]) + np.square(pts[5][1] - pts[6][1]))
        else:
            norm = np.sqrt(np.square(pts[15][0] - pts[16][0]) + np.square(pts[15][1] - pts[16][1]))
        if np.isnan(norm):
            print(' GT file not correct,  norm dis is NaN')
            exit(0)
        if norm == 0:
            norm = 256
        norm_mat[i] = norm
    return norm_mat

def calculate_norm_distance_mat(gt_data, pred_data, norm):
    samples = len(gt_data.keys())
    dis_mat = np.zeros((samples, 24))
    n = 0
    n_every_joints = np.zeros(24)
    for i, name in enumerate(gt_data.keys()):
        for j in range(24):
            # if gt_data[name]['joints'][j][2] != -1:
            if gt_data[name]['joints'][j][2] == 1:  ## only visible
                n += 1
                n_every_joints[j] += 1
                gt_pts = gt_data[name]['joints'][j]
                pre_pts = pred_data[name]['joints'][j]
                d = np.sqrt((gt_pts[0] - pre_pts[0]) * (gt_pts[0] - pre_pts[0]) + (gt_pts[1] - pre_pts[1]) * (
                gt_pts[1] - pre_pts[1]))
                dis_mat[i, j] = d / norm[i]
    return dis_mat, n, n_every_joints

def get_result_error(gt_data, predict_data):
    norm = calculate_norm(gt_data)
    norm_dis, N, n_every_joints = calculate_norm_distance_mat(gt_data, predict_data, norm)
    average_error = np.sum(norm_dis) / N * 100
    err_joints = np.sum(norm_dis, axis=0)
    err_joints = np.divide(err_joints, n_every_joints) * 100
    return average_error, err_joints

def fushion_two_result(raw_base_predict, base_predict1, base_predict2, filter_index):
    base_predict = copy.deepcopy(raw_base_predict)
    for key in base_predict:
        for i in data_attr[key.split("/")[1]]:
            i = i-2
            pre1_row = base_predict1[key]["joints"][i]
            pre2_row = base_predict2[key]["joints"][i]
            if i in filter_index:
                continue
            base_predict[key]["joints"][i] = (pre1_row + pre2_row) / 2
    return base_predict

def fushion_two_weight(raw_base_predict, base_predict1, base_predict2):
    base_predict = copy.deepcopy(raw_base_predict)
    for key in base_predict:
        for i in data_attr[key.split("/")[1]]:
            i = i-2
            weight = np.array([base_predict1[key]["joints"][i][2], base_predict2[key]["joints"][i][2]])
            # weight = np.sqrt(weight)
            # weight = np.square(weight)
            pre1_row = base_predict1[key]["joints"][i]
            pre2_row = base_predict2[key]["joints"][i]
            base_predict[key]["joints"][i] = (pre1_row*weight[0] + pre2_row*weight[1]) / (weight.sum())
    return base_predict

def find_filter_index(raw_base_predict, base_predict1, base_predict2):
    filter_index = []
    base_predict_original = copy.deepcopy(raw_base_predict)
    raw_predict_fushion = fushion_two_result(base_predict_original, base_predict1, base_predict2, filter_index)
    original_average_error, _ = get_result_error(gt_data, raw_predict_fushion)
    for i in range(24):
        base_predict = fushion_two_result(base_predict_original, base_predict1, base_predict2, [i])
        test_average_error, _ = get_result_error(gt_data, base_predict)
        if original_average_error > test_average_error:
            filter_index.append(i)
    return filter_index

def find_filter_weight_index(raw_base_predict, base_predict1, base_predict2):
    filter_index = []
    base_predict_original = copy.deepcopy(raw_base_predict)
    raw_predict_fushion = fushion_two_weight(base_predict_original, base_predict1, base_predict2, filter_index)
    original_average_error, _ = get_result_error(gt_data, raw_predict_fushion)
    for i in range(24):
        base_predict = fushion_two_weight(base_predict_original, base_predict1, base_predict2, [i])
        test_average_error, _ = get_result_error(gt_data, base_predict)
        if original_average_error > test_average_error:
            filter_index.append(i)
    return filter_index

def float_2_int_data(base):
    for key in base:
        for i in data_attr[key.split("/")[1]]:
            i = i-2
            temp = base[key]["joints"][i]
            base[key]["joints"][i] = [round(temp[0]), round(temp[1]), 1]

def normal_liuzhnehua_data(base):
    for key in base:
        for i in data_attr[key.split("/")[1]]:
            i = i-2
            temp = base[key]["joints"][i]
            base[key]["joints"][i] = [temp[0], temp[1], temp[2]/2]

if __name__ == '__main__':
    print("-------------val------------------")
    val_result = []
    val_base = read_data(annotations_path["val_base_predict2"])
    a = read_data(annotations_path["val_base_predict1"])  # liuzhenhua
    normal_liuzhnehua_data(a)
    b = read_data(annotations_path["val_base_predict2"])  # daiwei
    gt_data = read_data(annotations_path["gt"])

    average_error1, err_joints1 = get_result_error(gt_data, a)
    average_error2, err_joints2 = get_result_error(gt_data, b)
    val_result.append(round(average_error1, 3))
    val_result.append(round(average_error2, 3))
    print("base_predict1:", average_error1)
    print("base_predict2:", average_error2)
    ab_list = []

    if test_model == "test":
        test_base = read_data(annotations_path["test_base_predict2"])
        test_a = read_data(annotations_path["test_base_predict1"])
        test_b = read_data(annotations_path["test_base_predict2"])

    print("Weight method..........")
    val_base = fushion_two_weight(val_base, val_base, a)
    average_error, _ = get_result_error(gt_data, val_base)
    print(average_error)
    if test_model == "test":
        test_base = fushion_two_weight(test_base, test_base, test_a)

    print("Average method..........")
    for i in range(0):
        filter_index_1 = find_filter_index(val_base, val_base, a)
        base_predict_1 = fushion_two_result(val_base, val_base, a, filter_index_1)
        if test_model == "test":
            test_predict_1 = fushion_two_result(test_base, test_base, test_a, filter_index_1)
        average_error_1, _ = get_result_error(gt_data, base_predict_1)

        filter_index_2 = find_filter_index(val_base, val_base, b)
        base_predict_2 = fushion_two_result(val_base, val_base, b, filter_index_2)
        if test_model == "test":
            test_predict_2 = fushion_two_result(test_base, test_base, test_b, filter_index_2)
        average_error_2, _ = get_result_error(gt_data, base_predict_2)

        if average_error_1 < average_error_2:
            filter_index = filter_index_1
            val_base = base_predict_1
            if test_model == "test":
                test_base = test_predict_1
            average_error = average_error_1
            ab_list.append("a")
        else:
            filter_index = filter_index_2
            val_base = base_predict_2
            if test_model == "test":
                test_base = test_predict_2
            average_error = average_error_2
            ab_list.append("b")

        if len(filter_index) == 24:
            break
        print("filter_index——"+str(i)+":", filter_index)
        print("average_error:", average_error)

    print(ab_list)
    float_2_int_data(val_base)
    final_average_error, final_err_joints = get_result_error(gt_data, val_base)
    print("------------val_final----------------")
    print(final_average_error)

    # val_result.append(round(final_average_error, 3))
    # if test_model != "test":
    #     for i, v in enumerate(final_err_joints):
    #         print(v)

    if test_model == "test":
        print("-------------test------------------")
        val_result.append(round(final_average_error, 3))
        float_2_int_data(test_base)
        for key in test_base:
            row = ["-1_-1_-1" for i in range(26)]
            row[0] = key
            row[1] = test_base[key]["type"]
            for i in data_attr[test_base[key]["type"]]:
                point = test_base[key]["joints"][i - 2]
                s = str(int(point[0])) + "_" + str(int(point[1])) + "_1"
                row[i] = s
            result_csv.append(row)
        result_csv = np.array(result_csv)
        save_csv_path = "fushion_base1_" + str(val_result[0]) + "_base2_" + \
                        str(val_result[1]) + "_offline_" + str(val_result[2]) + "_oneline_unknown" + ".csv"
        print(save_csv_path)
        np.savetxt(os.path.join("submit", "fushion", save_csv_path), result_csv, fmt='%s', delimiter=',')