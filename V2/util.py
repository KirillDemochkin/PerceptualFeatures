import torch

def getNetImageSizeAndNumFeats(net, image_size = 32, use_cuda = False):
    '''return two list: 
    - list of size of output (image) for each layer
    - list of size of total number of features (nFeatMaps*featmaps_height,featmaps_width) '''
    if use_cuda:
        y, layers = net(torch.randn(1,3,image_size,image_size).cuda())
    else:
        y, layers = net(torch.randn(1,3,image_size,image_size))

    layer_img_size = []
    layer_num_feats = []
    for L in reversed(layers):
        if len(L.size()) == 4:
            layer_img_size.append(L.size(2))
            layer_num_feats.append(L.size(1)*L.size(2)*L.size(3))
        elif len(L.size()) == 2:
            layer_img_size.append(1)
            layer_num_feats.append(L.size(1))
        else:
            assert 0, 'not sure how to handle this layer size '+L.size()

    return layer_img_size, layer_num_feats


