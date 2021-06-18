from cnn import *


architecture = [
    {'layer_type': 'conv',      'filter_len': 2,        'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'layer_type': 'max_pool',  'pool_size': 2,    'stride': 2, 'padding': 0},
    
    {'layer_type': 'conv',      'filter_len': 8,        'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'layer_type': 'max_pool',  'pool_size': 2,    'stride': 2, 'padding': 0},

    {'layer_type': 'conv',      'filter_len': 8,        'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'layer_type': 'max_pool',  'pool_size': 2,    'stride': 2, 'padding': 0},

    {'layer_type': 'flatten'},
    
    {'layer_type': 'dense',     'input_size': 512,   'output_size': 256,   'activation': 'relu'},
    {'layer_type': 'dense',     'input_size': 256,   'output_size': 27,   'activation': 'relu'},
    {'layer_type': 'dense',     'input_size': 27,   'output_size': 4,   'activation': 'softmax'},
]

f = open('model.pckl', 'rb')
model = pickle.load(f)
f.close()

# x = w,x,y,z
# y = x,y,z
img1 = cv2.imread('dataset2/sepeda-motor.png', cv2.IMREAD_GRAYSCALE) / 255
img1 = cv2.resize(img1, (64, 64))
img2 = cv2.imread('dataset2/sepeda.png', cv2.IMREAD_GRAYSCALE) / 255
img2 = cv2.resize(img2, (64, 64))
img3 = cv2.imread('dataset2/mobil-penumpang.png', cv2.IMREAD_GRAYSCALE) / 255
img3 = cv2.resize(img3, (64, 64))
img4 = cv2.imread('dataset2/mobil-barang.png', cv2.IMREAD_GRAYSCALE) / 255
img4 = cv2.resize(img4, (64, 64))
x = np.array([
    np.array([img1]),
    np.array([img2]),
    np.array([img3]),
    np.array([img4]),
])
y = np.array([
    [[1], [0], [0], [0]],
    [[0], [1], [0], [0]],
    [[0], [0], [1], [0]],
    [[0], [0], [0], [1]]
])     
# y = np.reshape(y, (1, 4, 1))
cnn.train(np.array(tr_lbl_data), epochs=50)

