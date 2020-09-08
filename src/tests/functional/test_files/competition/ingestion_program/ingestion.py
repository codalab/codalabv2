import json
import os
import sys
import time

import numpy as np

input_dir = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])
program_dir = os.path.abspath(sys.argv[3])
submission_dir = os.path.abspath(sys.argv[4])
sys.path.append(program_dir)
sys.path.append(submission_dir)


def get_training_data():
    X_train = np.genfromtxt(os.path.join(input_dir, 'training_data'))
    y_train = np.genfromtxt(os.path.join(input_dir, 'training_label'))
    return X_train, y_train


def get_prediction_data():
    return np.genfromtxt(os.path.join(input_dir, 'testing_data'))


def main():
    from model import Model
    print('Reading Data')
    X_train, y_train = get_training_data()
    X_test = get_prediction_data()
    print('Starting')
    start = time.time()
    m = Model()
    print('Training Model')
    m.fit(X_train, y_train)
    print('Running Prediction')
    prediction = m.predict(X_test)
    duration = time.time() - start
    print(f'Completed Prediction. Total duration: {duration}')
    np.savetxt(os.path.join(output_dir, 'prediction'), prediction)
    with open(os.path.join(output_dir, 'metadata.json'), 'w+') as f:
        json.dump({'duration': duration}, f)


if __name__ == '__main__':
    main()
