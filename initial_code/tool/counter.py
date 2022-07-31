import torch

from model.pointtransformer.pointtransformer_cls_v5 import pointtransformer_cls26 as model
# from model.pointnet2.pointnet2_cls_v2 import PointNet2SSGCls as model

from util.flops_counter import get_model_complexity_info


with torch.cuda.device(0):
  # net = resnetmodels.resnet26v2(pretrained=False, num_classes=1000)
  # net = resnetmodels.resnet38(pretrained=False, num_classes=1000)
  # net = resnetmodels.resnet38(num_classes=1000)
  # net = resnetmodels.resnet26(num_classes=1000)
  # net = sanetmodels.sanet26(num_classes=1000)
  # net = sanetmodels.san(layers=[1, 1, 2, 4, 1]).cuda()
  # net = sanetmodels.san(layers=[2, 3, 4, 7, 2]).cuda()

  # net = sanetmodels.san(layers=[3, 1, 2, 4, 1], kernels=[3, 3, 3, 3, 3]).cuda()
  # net = sanetmodels.san(layers=[2, 1, 2, 4, 1], kernels=[3, 3, 3, 3, 3]).cuda()
  # net = sanetmodels.san(layers=[2, 1, 2, 4, 1], kernels=[3, 5, 5, 5, 5]).cuda()
  # net = sanetmodels.san(layers=[2, 1, 2, 4, 1], kernels=[3, 7, 7, 7, 7]).cuda()
  # net = sanetmodels.san(layers=[2, 1, 2, 4, 1], kernels=[3, 9, 9, 9, 9]).cuda()
  # net = sanetmodels.san(layers=[2, 1, 2, 4, 1], kernels=[3, 11, 11, 11, 11]).cuda()
  # net = sanetmodels.san(layers=[2, 2, 2, 4, 1], kernels=[3, 7, 7, 7, 7]).cuda()
  # net = sanetmodels.san(layers=[2, 2, 2, 5, 1], kernels=[3, 3, 3, 3, 3]).cuda()
  # net = sanetmodels.san(layers=[2, 2, 2, 5, 1], kernels=[3, 7, 7, 7, 7]).cuda()
  # net = sanetmodels.san(layers=[3, 3, 4, 6, 3], kernels=[3, 3, 3, 3, 3]).cuda()
  # net = sanetmodels.san(layers=[3, 3, 4, 6, 3], kernels=[3, 5, 5, 5, 5]).cuda()
  # net = sanetmodels.san(layers=[3, 3, 4, 6, 3], kernels=[3, 7, 7, 7, 7]).cuda()
  # net = sanetmodels.san(layers=[3, 4, 6, 8, 3], kernels=[3, 7, 7, 7, 7]).cuda()

  # net = sanetmodels.san(layers=[3, 2, 3, 5, 2], kernels=[3, 7, 7, 7, 7]).cuda()

  # net = senetmodels.senet50(pretrained=False, num_classes=1000)
  # net = resnextmodels.resnext101(pretrained=False, num_classes=1000)

  net = model(c=6, k=3).cuda()
  flops, params = get_model_complexity_info(net, (2048, 6), as_strings=True, print_per_layer_stat=True)
  print('Params/Flops: {}/{}'.format(params, flops))