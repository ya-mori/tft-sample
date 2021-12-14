import os
import shutil
from io import BytesIO
from urllib.request import urlopen

import apache_beam as beam
import numpy as np
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from PIL import Image
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils


tft_beam.Context.force_tf_compat_v1 = True


def fetch_image(inputs_vec):
    image_url = inputs_vec[0].decode()
    print(f"fetch_image called : {image_url}")
    pil_image = Image.open(BytesIO(urlopen(image_url).read()))
    np_image = np.array(pil_image, np.float32)  # shape=(150, 150)
    return np.expand_dims(np_image, axis=0)


def preprocessing_fn(inputs):
    # apply tft_function
    x = inputs['x']
    y = inputs['y']
    s = inputs['s']
    x_centered = x - tft.mean(x)
    y_normalized = tft.scale_to_0_1(y)
    s_integerized = tft.compute_and_apply_vocabulary(s)
    x_centered_times_y_normalized = x_centered * y_normalized

    # apply my_function  ※ pyfunc cannot be included in saved_model.pb.
    # image_vec = tft.apply_pyfunc(fetch_image, tf.float32, False, "hoge", inputs["image_url"])
    # image_vec.set_shape([1, 150, 150])

    return {
        'x_centered': x_centered,
        'y_normalized': y_normalized,
        'x_centered_times_y_normalized': x_centered_times_y_normalized,
        's_integerized': s_integerized,
        # "image_vec": image_vec,
    }


def preprocess(pipeline, output_dir):
    raw_data = [
        {'x': 1, 'y': 4, 's': 'hello', 'image_url': 'https://placehold.jp//ff0000/ffffff/150x150.png'},
        {'x': 2, 'y': 5, 's': 'new', 'image_url': 'https://placehold.jp//00ff00/ffffff/150x150.png'},
        {'x': 3, 'y': 6, 's': 'world', 'image_url': 'https://placehold.jp//0000ff/ffffff/150x150.png'}
    ]

    spec = {
        'y': tf.io.FixedLenFeature([], tf.float32),
        'x': tf.io.FixedLenFeature([], tf.float32),
        's': tf.io.FixedLenFeature([], tf.string),
        # 'image_url': tf.io.FixedLenFeature([], tf.string),
    }
    raw_data_metadata = dataset_metadata.DatasetMetadata(schema_utils.schema_from_feature_spec(spec))

    transformed_dataset, transform_fn = (
        (raw_data, raw_data_metadata)
        | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn)
    )
    transformed_data, transformed_metadata = transformed_dataset

    _ = (
        transformed_data
        | "WriteTrainData" >> beam.io.WriteToTFRecord(
            file_path_prefix=f"{output_dir}/tf_records/train",
            file_name_suffix=".tfrecord",
            coder=tft.coders.ExampleProtoCoder(transformed_metadata.schema),
        )
    )

    _ = (
        transform_fn
        | "WriteTransformFn" >> tft_beam.WriteTransformFn(
            path=output_dir
        )
    )

    _ = (
        transformed_metadata
        | "WriteMetadata" >> tft_beam.WriteMetadata(
            path=output_dir,
            pipeline=pipeline
        )
    )


def main():
    options = PipelineOptions()
    standard_options = options.view_as(StandardOptions)
    standard_options.runner = "DirectRunner"

    output_dir = "./data/output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shutil.rmtree(output_dir)

    with beam.Pipeline(options=options) as pipeline:
        # with tft_beam.Context(temp_dir=tempfile.mktemp()):
        with tft_beam.Context(temp_dir="./data/temp"):
            preprocess(pipeline, output_dir)


if __name__ == '__main__':
    main()
