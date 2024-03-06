import os
import io
import tensorflow as tf
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np


def create_tf_example(image_path, annotation_xml):
    # Load the image
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    # Parse the XML
    tree = ET.parse(annotation_xml)
    root = tree.getroot()

    # Initialize lists to store annotation data
    encoded_classes = []
    encoded_xmins = []
    encoded_ymins = []
    encoded_xmaxs = []
    encoded_ymaxs = []

    for image_tag in root.findall('.//image'):
        if image_tag.attrib['name'] == os.path.basename(image_path):
            for polygon in image_tag.findall('.//polygon'):
                label = polygon.attrib['label']
                # Here you can handle the label as needed, for example, encode it as an integer

                # Extract polygon points
                points = polygon.attrib['points'].split(';')
                points = [tuple(map(float, point.split(','))) for point in points]
                x_coords, y_coords = zip(*points)

                # Encode coordinates
                xmin = min(x_coords) / width
                xmax = max(x_coords) / width
                ymin = min(y_coords) / height
                ymax = max(y_coords) / height

                # Append encoded data to lists
                encoded_classes.append(label.encode('utf8'))
                encoded_xmins.append(xmin)
                encoded_ymins.append(ymin)
                encoded_xmaxs.append(xmax)
                encoded_ymaxs.append(ymax)

    # Convert lists to arrays
    encoded_classes = np.array(encoded_classes)
    encoded_xmins = np.array(encoded_xmins)
    encoded_ymins = np.array(encoded_ymins)
    encoded_xmaxs = np.array(encoded_xmaxs)
    encoded_ymaxs = np.array(encoded_ymaxs)

    # Create TFRecord example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=encoded_xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=encoded_xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=encoded_ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=encoded_ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_classes)),
    }))

    return tf_example


# Provide paths to XML annotation and image directory
annotation_xml = '/home/denis/PycharmProjects/Avto_ai_Bot/Data/Passport_JPG/Elements_Training/data_CVAT/annotations.xml'
image_dir = '/home/denis/PycharmProjects/Avto_ai_Bot/Data/Passport_JPG/Elements_Training/xlam'

# Create TFRecord for each image in the directory
tfrecord_filename = 'output.tfrecord'
with tf.io.TFRecordWriter(tfrecord_filename) as writer:
    tree = ET.parse(annotation_xml)
    root = tree.getroot()
    for image_tag in root.findall('.//image'):
        image_name = image_tag.attrib['name']
        image_path = os.path.join(image_dir, image_name)
        tf_example = create_tf_example(image_path, annotation_xml)
        writer.write(tf_example.SerializeToString())

print(f'TFRecord created: {tfrecord_filename}')
