import json

from utils import calculate_iou, check_results


def nms(predictions):
    """
    non max suppression
    args:
    - predictions [dict]: predictions dict 
    returns:
    - filtered [list]: filtered bboxes and scores
    """
    classes = predictions['classes']
    boxes = predictions['boxes']
    scores = predictions['scores']
    filtered =  [];
    
    
    for b_1,sc_1,cl_1 in zip(boxes,scores,classes):
        discard = False
        for b_2,sc_2 in zip(boxes,scores):
            if b_1 == b_2:
                continue
            if calculate_iou(b_1,b_2) > 0.5:
                if sc_2 >  sc_1   :
                    # Add the result into a set to make it unique
                    discard = True
        if not discard :
            
            filtered.append([b_1,sc_1])
    
    
    return filtered

if __name__ == '__main__':
    with open('data/predictions_nms.json', 'r') as f:
        predictions = json.load(f)
    
    filtered = nms(predictions)
    check_results(filtered)