""" Neural Style Transfer (Script Version)
This script applies the artistic style of one image onto another using TensorFlow Hub.
"""

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import PIL.Image

def load_img(path_to_img, max_dim=512):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / max(shape)

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def main():
    content_path = tf.keras.utils.get_file('content.jpg', 
        'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    style_path = tf.keras.utils.get_file('style.jpg', 
        'https://storage.googleapis.com/download.tensorflow.org/example_images/kandinsky5.jpg')

    content_image = load_img(content_path)
    style_image = load_img(style_path)

    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    result = tensor_to_image(stylized_image)
    result.save("stylized_image.jpg")
    print("Stylized image saved as 'stylized_image.jpg'.")

if __name__ == "__main__":
    main()
