class Config:
    tf_parallel_calls = 12
    img_slice_height = 512
    img_slice_width = 512
    smooth = 1e-5
    batch_size = 4
    train_epochs = 100
    prefetch_size = 16
    shuffle_size = 16
    folder_path = 'data'
    backbone_name = 'vgg19'
    backbone_layer_names = {
        'vgg19': [
            'block2_conv2',
            'block3_conv4',
            'block4_conv4',
            'block5_conv4',
            'block5_pool'],
        'resnet50': [
            'conv1_relu',
            'conv2_block3_out',
            'conv3_block4_out',
            'conv4_block6_out',
            'conv5_block3_out'],
        'resnet50v2': [
            'conv1_conv',
            'conv2_block3_1_relu',
            'conv3_block4_1_relu',
            'conv4_block6_1_relu',
            'post_relu'],
        'resnet101': [
            'conv1_relu',
            'conv2_block3_out',
            'conv3_block4_out',
            'conv4_block6_out',
            'conv5_block3_out'],
        'mobilenetv2': [
            'block_1_expand_relu',
            'block_3_expand_relu',
            'block_6_expand_relu',
            'block_13_expand_relu',
            'block_16_project']
    }