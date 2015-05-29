__author__ = 'cclamb'

import theano
import numpy

import theano.tensor as T


class NegativeLogLikelihood(object):

    def __init__(self, p_y_given_x):
        self._p_y_given_x = p_y_given_x

    def __call__(self, y):
        return -T.sum(T.log(self._p_y_given_x)[T.arange(y.shape[0]), y])


class ZeroOneLoss(object):

    def __init__(self, p_y_given_x):
        self._p_y_given_x = p_y_given_x

    def __call__(self, y):
        return T.sum(T.neq(T.argmax(self._p_y_given_x), y))


def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy

    shared_x = theano.shared(
        numpy.asarray(
            data_x,
            dtype=theano.config.floatX
        )
    )

    shared_y = theano.shared(
        numpy.asarray(
            data_y,
            dtype=theano.config.floatX
        )
    )

    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as floatX as well
    # (shared_y does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # shared_y we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')
