import random
import math

def mean_absolute_error(y_true, y_pred):
    return sum(abs(y_true[i] - y_pred[i]) for i in range(len(y_true))) / len(y_true)

def mean_squared_error(y_true, y_pred):
    return sum(abs(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)

def root_mean_squared_error(y_true , y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def generate_samples(num_samples):
    return [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_samples)]

def calculate_loss(num_samples, loss_name):
    if not num_samples.isnumeric():
        return 'number of samples must be an integer number'
    
    num_samples = int(num_samples)
    losses = []
    
    for i in range(num_samples):
        target = random.uniform(0, 10)
        prediction = random.uniform(0, 10)
        
        if loss_name == "MAE":
            loss = mean_absolute_error([target], [prediction])
        elif loss_name == "MSE":
            loss = mean_squared_error([target], [prediction])
        else:
            loss = root_mean_squared_error([target], [prediction])
        
        losses.append((f'loss_name: {loss_name}, sample: {i}, pred: {prediction}, target: {target}, loss: {loss}'))
    return losses

if __name__ == "__main__":
    num_samples = input('Input number of samples ( integer number ) which are generated :')
    loss_name = input('Input loss name :')
    
    losses = calculate_loss(num_samples, loss_name)
    for loss in losses:
        print(loss)