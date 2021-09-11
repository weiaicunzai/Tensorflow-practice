export CUDA_VISIBLE_DEVICES=0
python test.py -net resnet50 -gpu -b  2048 -weights /data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/ResNet50/97-best.pth
