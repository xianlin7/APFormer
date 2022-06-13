import matplotlib.pylab as plt
import torchvision
import os

def featuremap_visual(feature,
                      out_dir='./utils/visualization',  # 特征图保存路径文件
                      save_feature=True,  # 是否以图片形式保存特征图
                      show_feature=True,  # 是否使用plt显示特征图
                      feature_title=None,  # 特征图名字，默认以shape作为title
                      channel = None,
                      num_ch=-1,  # 显示特征图前几个通道，-1 or None 都显示
                      nrow=1,  # 每行显示多少个特征图通道
                      padding=10,  # 特征图之间间隔多少像素值
                      pad_value=1  # 特征图之间的间隔像素
                      ):

    # feature = feature.detach().cpu()
    b, c, h, w = feature.shape
    feature = feature[0]
    feature = feature.unsqueeze(1)
    if channel:
        feature = feature[channel]
    else:
        if c > num_ch > 0:
            feature = feature[:num_ch]

    img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
    img = img.detach().cpu()
    img = img.numpy()
    images = img.transpose((1, 2, 0))

    # title = str(images.shape) if feature_title is None else str(feature_title)
    title = str('hwc-') + str(h) + '-' + str(w) + '-' + str(c) if feature_title is None else str(feature_title)
    title = title + "_" + str(h) + '-' + str(w)

    plt.title(title)
    plt.imshow(images)
    if save_feature:
        # root=r'C:\Users\Administrator\Desktop\CODE_TJ\123'
        # plt.savefig(os.path.join(root,'1.jpg'))
        out_root = title + '.jpg' if out_dir == '' or out_dir is None else os.path.join(out_dir, title + '.jpg')
        plt.savefig(out_root)
    if show_feature:        plt.show()

