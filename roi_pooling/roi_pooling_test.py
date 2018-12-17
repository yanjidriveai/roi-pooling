import tensorflow as tf
import numpy as np
from roi_pooling_ops import roi_pooling


class RoiPoolingTest(tf.test.TestCase):
    # TODO(maciek): add python, implementation and test outputs
    # TODO(maciek): test pool_height != pool_width, height != width

    def test_roi_pooling_grad(self):
        # TODO(maciek): corner cases
        input_value = [[
            [[1], [2], [4], [4]],
            [[3], [4], [1], [2]],
            [[6], [2], [1], [7.0]],
            [[1], [3], [2], [8]]
        ]]
        input_value = np.asarray(input_value, dtype='float32')

        rois_value = [
            [0, 0, 0, 1, 1],
            [0, 1, 1, 2, 2],
            [0, 2, 2, 3, 3],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3]
        ]
        rois_value = np.asarray(rois_value, dtype='int32')

        with tf.Session(''):
            # NOTE(maciek): looks like we have to use consts here, based on tensorflow/python/ops/nn_test.py
            input_const = tf.constant(input_value, tf.float32)
            rois_const = tf.constant(rois_value, tf.int32)
            y = roi_pooling(input_const, rois_const, pool_height=2, pool_width=2)
            mean = tf.reduce_mean(y)

            numerical_grad_error_1 = tf.test.compute_gradient_error(
                [input_const], [input_value.shape], y, [5, 2, 2, 1])

            numerical_grad_error_2 = tf.test.compute_gradient_error(
                [input_const], [input_value.shape], mean, [])

            self.assertLess(numerical_grad_error_1, 1e-4)
            self.assertLess(numerical_grad_error_2, 1e-4)

    def test_very_big_output(self):
        """
        This test checks whether the layer can handle a corner case
        where the number of output pixels is very large, possibly larger
        than the number of available GPU threads
        """

        pooled_w, pooled_h = 7,7
        input_w, input_h = 72, 240
        n_channels = 512
        n_batches = 2
        x_input = np.ones(shape=(n_batches, input_w, input_h, n_channels))
        n_rois = 5000
        rois_input = np.ones(shape=(n_rois, 5))

        input = tf.placeholder(tf.float32, shape=[n_batches, input_w, input_h, n_channels])
        single_roi_dimension = 5
        rois = tf.placeholder(tf.int32, shape=[n_rois, single_roi_dimension])

        y = roi_pooling(input, rois, pool_height=pooled_w, pool_width=pooled_h)

        with tf.Session('') as sess:
            y_output = sess.run(y, feed_dict={input: x_input, rois: rois_input})

        self.assertTrue(np.all(y_output == 1))

if __name__ == '__main__':
    tf.test.main()
