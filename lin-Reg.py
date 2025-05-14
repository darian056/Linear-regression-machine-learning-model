from numpy import *
import matplotlib.pyplot as plt

def compute_error_for_line_given_points(b, m, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def gradient_descent_runner(points, start_b, start_m, learning_rate, num_iterations):
    b = start_b
    m = start_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))
    for i in range(int(n)):  
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / n) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / n) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def run():
    points = genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    print('starting gradient descent at b = {0}, m = {1}, error = {2}'.format(
        initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print('after {0} iterations: b = {1}, m = {2}, error = {3}'.format(
        num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
    
    x = points[:, 0]
    y = points[:, 1]
    predicted_y = m * x + b

    plt.scatter(x, y)
    plt.plot(x, predicted_y)
    plt.xlabel('number of hours studied')
    plt.ylabel('score')
    plt.title('linear regression learning model')
    plt.show()

if __name__ == '__main__':
    run()
