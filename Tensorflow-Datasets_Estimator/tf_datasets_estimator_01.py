# Example: https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html

import tensorflow as  tf

import os
ROOT = os.path.dirname(__file__)
DATAPATH = os.path.join(ROOT, 'data')
FILE = os.path.join(DATAPATH, 'iris_training.csv')

# Create a list to be used in feature columns:
FEATURE_NAMES = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth']


# Estimators requires an input_fn()

def input_fn(file_path, shuffle=False, repeat_count=1):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, record_defaults=[[0.]]*5)
        label = parsed_line[-1:]
        features = parsed_line[:-1]
        return dict(zip(FEATURE_NAMES, features)), label

    # Creates an instace of the TextLineDataset class
    dataset = (tf.data.TextLineDataset(file_path).skip(1).map(decode_csv))
    
    if shuffle:
       # Randomizes input using a window of 256 elements (read into memory)
       dataset = dataset.shuffle(buffer_size=256)
        
    dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
    dataset = dataset.batch(8)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    
    return batch_features, batch_labels
    

if __name__ == "__main__":
    next_batch = input_fn(FILE, True) # Will return 32 random elements

    with tf.Session() as sess:
        first_batch = sess.run(next_batch)
        
    print(first_batch)