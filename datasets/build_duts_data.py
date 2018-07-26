import math
import os.path
import sys
import build_data
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_image_folder',
                           '/data/zhbli/Dataset/DUTS/DUTS-TR/DUTS-TR-Image',
                           'Folder containing images.')

tf.app.flags.DEFINE_string('test_image_folder',
                           '/data/zhbli/Dataset/DUTS/DUTS-TE/DUTS-TE-Image',
                           'Folder containing images.')

tf.app.flags.DEFINE_string(
    'train_semantic_segmentation_folder',
    '/data/zhbli/Dataset/DUTS/DUTS-TR/DUTS-TR-Mask',
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'test_semantic_segmentation_folder',
    '/data/zhbli/Dataset/DUTS/DUTS-TE/DUTS-TE-Mask',
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'train_superpixel_folder',
    '/data/zhbli/Dataset/DUTS/DUTS-TR/DUTS-TR-Superpixel',
    'Folder containing super pixel maps.')

tf.app.flags.DEFINE_string(
    'test_superpixel_folder',
    '/data/zhbli/Dataset/DUTS/DUTS-TE/DUTS-TE-Superpixel',
    'Folder containing super pixel maps.')

tf.app.flags.DEFINE_string(
    'list_folder',
    '/data/zhbli/Dataset/DUTS/seg_list',
    'Folder containing lists for training and testing')

tf.app.flags.DEFINE_string(
    'output_dir',
    '/data/zhbli/Dataset/DUTS/tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')

_NUM_SHARDS = 4


def _convert_dataset(dataset_split):
    dataset = os.path.basename(dataset_split)[:-4]  # train or test
    print('Processing ' + dataset)
    filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
    num_images = len(filenames)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = build_data.ImageReader('jpeg', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)
    superpixel_reader = build_data.ImageReader('png', channels=1)

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            FLAGS.output_dir,
            '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                print('\r>> Converting image %d/%d shard %d' % (
                    i + 1, len(filenames), shard_id))
                if dataset == 'train':
                    image_filename = os.path.join(
                        FLAGS.train_image_folder, filenames[i] + '.' + FLAGS.image_format)
                    seg_filename = os.path.join(
                        FLAGS.train_semantic_segmentation_folder,
                        filenames[i] + '.' + FLAGS.label_format)
                    superpixel_filename = os.path.join(
                        FLAGS.train_superpixel_folder,
                        filenames[i] + '.' + FLAGS.label_format)
                elif dataset == 'test':
                    image_filename = os.path.join(
                        FLAGS.test_image_folder, filenames[i] + '.' + FLAGS.image_format)
                    seg_filename = os.path.join(
                        FLAGS.test_semantic_segmentation_folder,
                        filenames[i] + '.' + FLAGS.label_format)
                    superpixel_filename = os.path.join(
                        FLAGS.test_superpixel_folder,
                        filenames[i] + '.' + FLAGS.label_format)
                else:
                    assert False

                # Read the image.
                image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
                height, width = image_reader.read_image_dims(image_data)
                # Read the semantic segmentation annotation.
                seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)
                # Read the super pixels.
                superpixel_data = tf.gfile.FastGFile(superpixel_filename, 'rb').read()
                sp_height, sp_width = superpixel_reader.read_image_dims(superpixel_data)
                if not (height == seg_height == sp_height and width == seg_width == sp_width):
                    raise RuntimeError('Shape mismatched between image, label and super pixel map.')

                # Convert to tf example.
                example = build_data.image_seg_sp_to_tfexample(image_data, filenames[i], height, width, seg_data, superpixel_data)
                tfrecord_writer.write(example.SerializeToString())
        print('\n')


def main(unused_argv):
  dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*.txt'))
  for dataset_split in dataset_splits:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()