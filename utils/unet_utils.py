import tensorflow as tf
from config import Config
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from utils.dataset_utils import ImageViewer


class Model():
    def __init__(self):
        pass

    def unindex(self, image, mask, image_path):
        return image, mask

    def create_backbone(self, name='vgg19', set_trainable=True):
        """ Creates a backbone for segmentation model.
            Args:
            name: either: 'vgg19', 'resnet50', 'resnet50v2', 'mobilenetv2', 'resnet101'
            set_trainable: either; True or False
            Returns:
            tf.keras functional model
        """
        if name == 'vgg19':
            backbone = tf.keras.applications.VGG19(input_shape=[Config.img_slice_height, Config.img_slice_width, 3],
                                                   include_top=False)
        elif name == 'resnet50':
            backbone = tf.keras.applications.ResNet50(input_shape=[Config.img_slice_height, Config.img_slice_width, 3],
                                                      include_top=False)
        elif name == 'resnet50v2':
            backbone = tf.keras.applications.ResNet50V2(
                input_shape=[Config.img_slice_height, Config.img_slice_width, 3],
                include_top=False)
        elif name == 'mobilenetv2':
            backbone = tf.keras.applications.MobileNetV2(
                input_shape=[Config.img_slice_height, Config.img_slice_width, 3],
                include_top=False)
        elif name == 'resnet101':
            backbone = tf.keras.applications.ResNet101(input_shape=[Config.img_slice_height, Config.img_slice_width, 3],
                                                       include_top=False)
        else:
            raise ValueError('No Backbone for Name "{}" defined \nPossible Names are: {}'.format(name, list(
                Config.backbone_layer_names.keys())))
        backbone.trainable = set_trainable
        return backbone

    def segmentation_model_func(self, output_channels, backbone_name, backbone_trainable=True):
        """ Creates a segmentation model with the tf.keras functional api.
            Args:
            output_channels: number of output_channels (classes)
            backbone_name: name of backbone; either: 'vgg19', 'resnet50', 'resnet50v2', 'mobilenetv2', 'resnet101'
            Returns:
            tf.keras functional model
        """
        down_stack = self.create_backbone(name=backbone_name, set_trainable=backbone_trainable)

        skips = [down_stack.get_layer(Config.backbone_layer_names[backbone_name][0]).output,
                 down_stack.get_layer(Config.backbone_layer_names[backbone_name][1]).output,
                 down_stack.get_layer(Config.backbone_layer_names[backbone_name][2]).output,
                 down_stack.get_layer(Config.backbone_layer_names[backbone_name][3]).output,
                 down_stack.get_layer(Config.backbone_layer_names[backbone_name][4]).output]

        up_stack_filters = [32, 64, 128, 256]

        x = skips[-1]
        skips = reversed(skips[:-1])
        up_stack_filters = reversed(up_stack_filters)

        # Upsampling and establishing the skip connections
        for skip, filters in zip(skips, up_stack_filters):
            x = self.ublock(x, filters, 3, 'up_stack' + str(filters))
            x = tf.keras.layers.Concatenate()([x, skip])

        # x = simple_upblock_func(x, 32, 3, 'up_stack' + str(32))
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        if output_channels == 1:
            x = tf.keras.layers.Conv2D(output_channels, 1, activation='sigmoid', padding='same', name='final_output')(x)
        else:
            x = tf.keras.layers.Conv2D(output_channels, 1, activation='softmax', padding='same', name='final_output')(x)

        return tf.keras.Model(inputs=down_stack.layers[0].input, outputs=x)

    def ublock(self, input_layer, filters, size, block_name, norm_type='batchnorm', apply_dropout=False):
        """ Upsamples an input.
            Conv2DTranspose => Batchnorm => Dropout => Relu
            Args:
            input_layer: input layer to apply upsampling
            filters: number of filters
            size: filter size
            norm_type: Normalization type; 'batchnorm'.
            apply_dropout: If True, adds the dropout layer
            Returns:
            tf.keras functional layer
        """
        x = tf.keras.layers.UpSampling2D(2, name=block_name)(input_layer)

        x = tf.keras.layers.Conv2D(filters, size, padding='same')(x)

        if norm_type.lower() == 'batchnorm':
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters, size, padding='same')(x)

        if norm_type.lower() == 'batchnorm':
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.ReLU()(x)

        if apply_dropout:
            x = tf.keras.layers.Dropout(0.3)(x)

        return x

    def get_dice_score(self, msk, pred, skip_background=True):
        """ Dice Score Metric for Training and Validation.
            Args:
            msk: ground truth mask [batchsize, height, width, classes], type bool
            pred: prediction mask [batchsize, height, width, classes], type bool
            skip_background: if skipping last class (background) for calculation
            Returns:
            dice scalar
        """
        if skip_background:
            msk = msk[..., 0:2]
            pred = pred[..., 0:2]

        batch_size = msk.shape[0]
        metric = []

        for batch in range(batch_size):
            m, p = msk[batch], pred[batch]
            intersection = np.logical_and(m, p)
            denominator = np.sum(m) + np.sum(p)
            if denominator == 0.0:
                denominator = np.finfo(float).eps
            dice_score = 2. * np.sum(intersection) / denominator
            metric.append(dice_score)

        return np.mean(metric)

    def my_dice_metric_foreground(self, label, pred):
        """ Converts dice score metric to tensorflow graph, only hemp
            Args:
            label: ground truth mask [batchsize, height, width, classes]
            pred: prediction mask [batchsize, height, width, classes]
            Returns:
            dice value as tensor
        """
        return tf.py_function(self.get_dice_score, [label > 0.5, pred > 0.5], tf.float32)

    def my_dice_metric_all(self, label, pred):
        """ Converts dice score metric to tensorflow graph, all classes
            Args:
            label: ground truth mask [batchsize, height, width, classes]
            pred: prediction mask [batchsize, height, width, classes]
            Returns:
            dice value as tensor
        """
        return tf.py_function(self.get_dice_score, [label > 0.5, pred > 0.5, False], tf.float32)
    
    def dice_loss(self, y_true, y_pred):
        return 1 - self.dice_score(y_true=y_true, y_pred=y_pred)

    def jaccard_distance_loss(self, y_true, y_pred, smooth=100):
        return (1 - self.jaccard_distance(y_true=y_true, y_pred=y_pred, smooth=smooth)) * smooth

    def dice_score(self, y_true, y_pred, skip_background=False):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        return numerator / denominator

    def jaccard_distance(self, y_true, y_pred, smooth=100):
        """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
        The jaccard distance loss is usefull for unbalanced datasets. This has been
        shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
        gradient.
        Ref: https://en.wikipedia.org/wiki/Jaccard_index
        """
        intersection = tf.math.reduce_sum(tf.abs(y_true * y_pred), axis=-1)
        sum_ = tf.math.reduce_sum(tf.abs(y_true) + tf.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return jac


class Train(Model):
    def __init__(self, train_set, val_set):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.model = None
        self._learning_rate_schedule = tf.keras.callbacks.LearningRateScheduler(StepDecay(initAlpha=Config.init_learning_rate, factor=0.3,
                                                                                          dropEvery=20))
        self._model_checkpoint = tf.keras.callbacks.ModelCheckpoint('models/best_{}.h5'.format(Config.backbone_name),
                                                                    monitor='val_my_dice_metric_all',
                                                                    mode='max', save_best_only=True, verbose=1)
        self._tensorboard_log = tf.keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=1, write_graph=True,
                                                               write_images=True, update_freq='epoch', profile_batch=2,
                                                               embeddings_freq=1, embeddings_metadata=None)
        self._image_saver = ImageSaver(train_set_copy=train_set, val_set_copy=val_set)

    def create_model(self, output_channels, backbone_name, backbone_trainable):
        self.model = self.segmentation_model_func(output_channels=output_channels,
                                                  backbone_name=backbone_name,
                                                  backbone_trainable=backbone_trainable)
        self._compile_model()

    def fit(self, train_set_size):
        model_history = self.model.fit(self.train_set.map(self.unindex).repeat().prefetch(Config.prefetch_size),
                                       epochs=Config.train_epochs,
                                       steps_per_epoch=(train_set_size // Config.batch_size)*2,
                                       validation_steps=None,
                                       validation_data=self.val_set.map(self.unindex).prefetch(Config.prefetch_size),
                                       callbacks=[self._learning_rate_schedule,
                                                  self._model_checkpoint,
                                                  self._tensorboard_log,
                                                  self._image_saver])
        return model_history

    def _compile_model(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=self.dice_loss,
                           metrics=[self.my_dice_metric_all])


class Predict(Model):
    def __init__(self):
        super().__init__()
        raise NotImplementedError()


class StepDecay():
    """ Creates a learning rate Step Decay callback for training
        Args:
        initAlpha: initial learning rate
        factor: factor by which to multiply the learning rate after every drop
        dropEvery: Epochs to drop the learning rate
        Returns:
        Learning rate based on settings and current epoch
    """

    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)

        # return the learning rate
        print(" Learning Rate: {}".format(float(alpha)))
        tf.summary.scalar('learning rate', data=float(alpha))
        return float(alpha)


class ImageSaver(tf.keras.callbacks.Callback):
    def __init__(self, train_set_copy, val_set_copy, n=20):
        super().__init__()
        self.train_set_copy = train_set_copy
        self.val_set_copy = val_set_copy
        self.n = n
        logdir = "logs/train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_writer = tf.summary.create_file_writer(logdir)

    def _overlay_pair(self, img, msk):
        msk *= 0.4

        # make overlay
        color = (1, 0, 0.4)
        overlay = np.ones(img.shape, dtype=np.float) * color

        # overlay over original image
        out = overlay * msk + img * (1.0 - msk)
        return np.expand_dims(out, axis=0)

    def on_epoch_begin(self, epoch, logs=None):
        output = list(self.train_set_copy.take(self.n))
        for i, (sample_img, sample_msk, img_path) in enumerate(output):
            image = self._overlay_pair(img=sample_img.numpy()[0, :],
                                       msk=sample_msk.numpy()[0, :])
            with self.file_writer.as_default():
                tf.summary.image('Training data', image, step=i)


class MetricLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        pass


def display(image, mask, prediction=None):
    if prediction is None:
        _, ax = plt.subplots(1, 2, figsize=(15, 15))
    else:
        _, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(image)
    ax[0].set_title('image')
    ax[0].axis('off')
    ax[1].imshow(mask)
    ax[1].set_title('mask')
    ax[1].axis('off')
    if prediction is not None:
        ax[2].imshow(prediction)
        ax[2].set_title('prediction')
        ax[2].axis('off')
    plt.tight_layout()
