import tensorflow as tf
from tensorflow.keras.preprocessing import image

def classification(img):    
    x = image.img_to_array(img)
    x = x.astype('float32')
    x = x.reshape((1,) + x.shape)
    x /= 255

    model = tf.keras.models.load_model('resnet_model')
    result = model.predict(x)[0][0]

    return result