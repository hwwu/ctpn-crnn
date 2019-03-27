# cptn-crnn

## cptn

1.环境搭建(tqdm,opencv-python,Shapely,matplotlib,numpy,tensorflow-gpu or tensorflow,Cython,ipython 自行用pip3安装)
 cd cptn/utils/bbox
 sh make.sh

2.创建数据集
 cd cptn/utils/prepare
 sh split_label.py(DATA_FOLDER 和 OUTPUT 改成自己的路径)

 原始数据图片如下：
 ![Image text](https://github.com/hwwu/cptn-crnn/blob/master/cptn/data/demo/source/img_calligraphy_70001_bg.jpg)

