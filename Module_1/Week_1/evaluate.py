def evaluate_model(tp, fp, fn):
    
    if type(tp) != int or type(fp) != int or type(fn) != int:
        if type(tp) != int:
            return 'tp must be int'
        elif type(fp) != int:
            return 'fp must be int'
        else:
            return 'fn must be int'
        
    if tp <= 0 or fp <= 0 or fn <= 0:
        return 'tp and fp and fn must be greater than zero'
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    
    precision_str = f'precision is {precision}'
    recall_str = f'recall is {recall}'
    f1_score_str = f'f1_score is {f1_score}'
    
    return precision_str, recall_str, f1_score_str