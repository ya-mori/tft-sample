import tensorflow as tf
import tensorflow_transform as tft


def main():
    tft_output = tft.TFTransformOutput("./data/output")
    request = {
        'x': tf.constant(1, tf.float32, [1]),
        'y': tf.constant(1, tf.float32, [1]),
        's': tf.constant('hello', tf.string, [1]),
        'image_url': tf.constant(
            'https://placehold.jp//000000/ffffff/150x150.png',
            tf.string,
            [1]
        ),
    }
    features = tft_output.transform_raw_features(request)
    print(features)


if __name__ == '__main__':
    main()
