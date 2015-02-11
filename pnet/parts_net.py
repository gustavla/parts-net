from __future__ import division, print_function, absolute_import

import numpy as np
import amitgroup as ag
from pnet.layer import Layer
import pnet


class ExtractionFunction:
    """
    Extracting feature from a certain layer, means that layer
    can call extract on the previous layer, which in turn can
    call on the previous, etc. Since the layers aren't aware of
    one another, this class acts as an extraction function
    instead.
    """
    def __init__(self, layer=None, phi=None, pos_matrix=None, name='Identity'):
        self.layer = layer
        self.phi = phi
        self.pos_matrix = pos_matrix
        self.name = name
        self._cache = {}

    def __call__(self, X):
        return X

    def __repr__(self):
        return 'ExtractionFunction<{}>'.format(self.name)


def smart_hash(x):
    if isinstance(x, np.ndarray):
        return hash(x.tostring())
    elif isinstance(x, tuple):
        return hash(tuple([smart_hash(child) for child in x]))
    else:
        return hash(x)


@Layer.register('parts-net')
class PartsNet(Layer):
    def __init__(self, layers):
        self._layers = layers
        self._train_info = {}
        self._trained = False
        self.caching = False
        self._extract_funcs = []

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, value):
        self._layers = value
        self._prepare_extract_funcs()
        return self._layers

    @property
    def trained(self):
        return self._trained

    def _prepare_extract_funcs(self):
        # This does not work if someone has removed and added layers so that
        # the end count is the same.
        if len(self._extract_funcs) == 1 + len(self._layers):
            return

        fs = [ExtractionFunction()]
        As = [pnet.matrix.identity()]
        for i, layer in enumerate(self._layers):
            parent_self = self

            A = np.dot(layer.pos_matrix, As[-1])

            class F(ExtractionFunction):
                def __call__(self, x):
                    if parent_self.caching:
                        h = smart_hash(x)
                        if h in self._cache:
                            return self._cache[h]

                    v = self.layer._extract(self.phi, x)

                    if parent_self.caching:
                        self._cache[h] = v
                    return v

            f = F(layer=layer,
                  phi=fs[-1],
                  pos_matrix=A,
                  name='{}:{}'.format(i, layer.name))

            fs.append(f)
            As.append(A)

        # We don't need to keep the identity one around, so we'll remove that
        # before storing it
        self._extract_funcs = fs

    def train(self, data, y=None):
        return self._train(lambda x: x, data, y=y)

    def _train(self, phi, data, y=None):
        X = phi(data)
        self._prepare_extract_funcs()

        old_caching = self.caching
        self.caching = False  # TODO: Until this can be better managed

        # Layers that need training and can be given training (this rules out
        # unsupervised if y is not specified). We fetch these so that we can
        # determine how far up we need to extract features.
        def needs_supervised(layer):
            return (not layer.supervised or
                    (layer.supervised and y is not None))

        def check_needs(layer):
            return not layer.trained and needs_supervised(layer)

        needs_training = [check_needs(layer) for layer in self._layers]
        needs_training_indices = np.where(needs_training)[0]
        if len(needs_training_indices) > 0:
            max_layer = needs_training_indices[-1] + 1
        else:
            max_layer = 0
        layers = self._layers[:max_layer]

        for l, layer in enumerate(layers):
            if not needs_supervised(layer):
                break

            if not layer.trained:
                ag.info('Training layer {}...'.format(l))
                layer._train(self._extract_funcs[l], X, y=y)
                ag.info('Done.')

        # Here There

        self.caching = old_caching

        self._trained = True

    def test(self, X, y):
        yhat = self.classify(X)
        return yhat == y

    def classify(self, X):
        return self.extract(X, classify=True)

    def first_classifier_index(self):
        is_classifier = [layer.classifier for layer in self.layers]
        try:
            return is_classifier.index(True)
        except ValueError:
            return len(self.layers)

    def extract(self, data, classify=False, layer=None):
        return self._extract(lambda x: x, data, classify=classify, layer=layer)

    def _extract(self, phi, data, classify=False, layer=None):
        X = phi(data)
        if layer is None:
            index = len(self.layers)-1 if classify \
                else self.first_classifier_index() - 1
        else:
            index = layer
        self._prepare_extract_funcs()
        return self._extract_funcs[index + 1](X)

    def _vzlog_output_(self, vz):
        vz.title('Layers')

        if 0:
            vz.text('Layer shapes')
            shapes = self._train_info['shapes']
            for i, (layer, shape) in enumerate(zip(self.layers, shapes)):
                vz.log(i, shape, layer.name)

        # Plot some information
        for i, layer in enumerate(self.layers):
            vz.section('Layer #{} : {}'.format(i, layer.name))
            vz.output(layer)

    def save_to_dict(self):
        d = {}
        d['layers'] = []
        for layer in self._layers:
            layer_dict = layer.save_to_dict()
            layer_dict['name'] = layer.name
            d['layers'].append(layer_dict)

        return d

    @classmethod
    def load_from_dict(cls, d):
        # Build the layers from the dictionary
        layers = [
            Layer.getclass(layer_dict['name']).load_from_dict(layer_dict)
            for layer_dict in d['layers']
        ]
        obj = cls(layers)
        return obj

    def __repr__(self):
        return 'PartsNet(n_layers={n_layers})'.format(
               n_layers=len(self._layers))
