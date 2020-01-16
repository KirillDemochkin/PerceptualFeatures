def inference_on_directory(directory):
    # call inference function to get a list of results
    '''

    [
        [
            {"top_left": [x1, y1], "bottom_right": [x2, y2]},
            {"top_left": [x1, y1], "bottom_right": [x2, y2]},
            {"top_left": [x1, y1], "bottom_right": [x2, y2]}
        ],
        [
            {"top_left": [x1, y1], "bottom_right": [x2, y2]},
            {"top_left": [x1, y1], "bottom_right": [x2, y2]},
            {"top_left": [x1, y1], "bottom_right": [x2, y2]}
        ],
    ]

    '''
    pass

def load_ground_truth_labels(dataset):
    # load the ground truth labels in the following format
    '''

    [
        [
            {"top_left": [x1, y1], "bottom_right": [x2, y2]},
            {"top_left": [x1, y1], "bottom_right": [x2, y2]},
            {"top_left": [x1, y1], "bottom_right": [x2, y2]}
        ],
        [
            {"top_left": [x1, y1], "bottom_right": [x2, y2]},
            {"top_left": [x1, y1], "bottom_right": [x2, y2]},
            {"top_left": [x1, y1], "bottom_right": [x2, y2]}
        ],
    ]

    '''


def bb_intersection_over_union(bbox1_top_left, bbox1_bottom_right, bbox2_top_left, bbox2_bottom_right):
    xA = max(bbox1_top_left[0], bbox2_top_left[0])
    yA = max(bbox1_top_left[1], bbox2_top_left[1])
    xB = min(bbox1_bottom_right[0], bbox2_bottom_right[0])
    yB = min(bbox1_bottom_right[1], bbox2_bottom_right[1])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (bbox1_bottom_right[0] - bbox1_top_left[0] + 1) * (bbox1_bottom_right[1] - bbox1_top_left[1] + 1)
    boxBArea = (bbox2_bottom_right[0] - bbox2_top_left[0] + 1) * (bbox2_bottom_right[1] - bbox2_top_left[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def calculate_precision_recall(directory, iou_threshold):
    #detections = inference_on_directory(directory)
    #ground_truths = load_ground_truth_labels('dataset_path')
    detections = [
        [
            {"top_left": [1, 1], "bottom_right": [4, 4]},
            {"top_left": [0, 7], "bottom_right": [2, 9]},

        ]
    ]

    ground_truths = [
        [
            {"top_left": [2, 2], "bottom_right": [5, 5]},
            {"top_left": [3, 7], "bottom_right": [5, 9]},

        ]
    ]
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(len(detections)):
        # check true positives
        found = [False for _ in range(len(ground_truths[i]))]
        for box in detections[i]:
            fp = True
            for j in range(len(ground_truths[i])):
                gbox = ground_truths[i][j]
                if bb_intersection_over_union(box["top_left"], box["bottom_right"], gbox["top_left"], gbox["bottom_right"]) >= iou_threshold:
                    found[j] = True
                    fp = False
                    break
            if fp:
                false_positives += 1
            else:
                true_positives += 1
        for find_result in found:
            if not find_result:
                false_negatives += 1

    precision = true_positives/(true_positives+false_positives)
    recall = true_positives/(true_positives+false_negatives)
    return precision, recall

p, r = calculate_precision_recall('some_dir', 0.1)
print(p, r)