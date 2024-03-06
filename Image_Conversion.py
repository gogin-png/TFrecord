import tensorflow as tf
from object_detection.utils import dataset_util

def create_tf_example(image_path, boxes, classes):
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()

    width, height =  # получить ширину и высоту изображения

    filename = image_path.split('/')[-1].encode('utf8')
    image_format = b'jpg'  # формат изображения

    xmins = []  # список с координатами x минимального угла bounding box
    xmaxs = []  # список с координатами x максимального угла bounding box
    ymins = []  # список с координатами y минимального угла bounding box
    ymaxs = []  # список с координатами y максимального угла bounding box
    classes_text = []  # список с текстовыми метками классов объектов
    classes = []  # список с числовыми метками классов объектов

    for box, class_id in zip(boxes, classes):
        xmins.append(box[0] / width)
        xmaxs.append(box[2] / width)
        ymins.append(box[1] / height)
        ymaxs.append(box[3] / height)
        classes_text.append(class_id.encode('utf8'))
        classes.append(class_id)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example

# Пример использования
def main(_):
    writer = tf.io.TFRecordWriter('output.tfrecord')
    image_path = 'path/to/your/image.jpg'
    boxes = [[xmin, ymin, xmax, ymax]]  # список bounding boxes для каждого изображения
    classes = ['class1', 'class2']  # список классов объектов для каждого изображения

    tf_example = create_tf_example(image_path, boxes, classes)
    writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    tf.compat.v1.app.run()
