#### Code Repository for the CVPR 2018 DeepGlobe Workshop paper [Semantic Binary Segmentation using Convolutional Networks without Decoders](https://arxiv.org/abs/1805.00138)

#### [Pretrained Models](https://drive.google.com/open?id=1M16yUTzUu0esbcYIhL5SgOc4ge50L6a9)

* _(train resnet50 oneway mdoel)_ python main\_resnet\_d2s.py --train 1 --data 'TRAIN\_IMG\_DIR' --gt 'TRAIN\_MASK\_DIR' --arch 'resnet50' --optim 'adam' --workers 8 --batch-size 8 --start-epoch 1 --end-epoch 100 --learning-rate 0.0001 --momentum 0.9 --weight-decay 0.0001 --save-interval 1 --print-freq 10

* _(train vgg16-bn oneway mdoel)_ python main\_vgg\_d2s.py --train 1 --data 'TRAIN\_IMG\_DIR' --gt 'TRAIN\_MASK\_DIR' --arch 'vgg16\_bn' --optim 'adam' --workers 8 --batch-size 8 --start-epoch 1 --end-epoch 100 --learning-rate 0.0001 --momentum 0.9 --weight-decay 0.0001 --save-interval 1 --print-freq 10

* _(train segnet oneway mdoel)_ python main\_segnet.py --train 1 --data 'TRAIN\_IMG\_DIR' --gt 'TRAIN\_MASK\_DIR' --arch 'segnet' --optim 'adam' --workers 8 --batch-size 8 --start-epoch 1 --end-epoch 100 --learning-rate 0.0001 --momentum 0.9 --weight-decay 0.0001 --save-interval 1 --print-freq 10

* _(test mdoel)_ python main\_XXXX.py --train 0 --data 'TEST\_IMG\_DIR' --out 'OUT\_MASK\_DIR' --arch XXXX --workers 1 --load-epoch XX
