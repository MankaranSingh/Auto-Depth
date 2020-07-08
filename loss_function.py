import tensorflow as tf
import keras.backend as K

# Berhu loss works better than vanilla mae in this case.
def berhu_loss(labels, predictions, scope=None):

    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    abs_error = tf.abs(tf.subtract(predictions, labels), name='abs_error')
    c = 0.2 * tf.reduce_max(abs_error)

    berHu_loss = tf.where(abs_error <= c,   
                  abs_error, 
                  (tf.square(abs_error) + tf.square(c))/(2*c))
            
    loss = tf.reduce_mean(berHu_loss)

    return loss


# Taken from original DenseDepth Repo
def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=100):
    
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
    l_ssim = (1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5

    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))
