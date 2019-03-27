# cptn-crnn

从别人那里拿来的，稍微改了下，适合识别竖排书法，原文：

https://github.com/eragonruan/text-detection-ctpn

https://github.com/Sierkinhane/crnn_chinese_characters_rec


## cptn

### 环境搭建(tqdm,opencv-python,Shapely,matplotlib,numpy,tensorflow-gpu or tensorflow,Cython,ipython 自行用pip3安装)
 cd cptn/utils/bbox
 sh make.sh

### 创建数据集
 cd cptn/utils/prepare
 sh split_label.py(DATA_FOLDER 和 OUTPUT 改成自己的路径)

 原始数据图片：
 ![Image text](https://github.com/hwwu/cptn-crnn/blob/master/cptn/data/demo/source/img_calligraphy_70001_bg.jpg)
 
 csv标记内容为：
 
  FileName                    | x1| y1| x2| y2| x3| y3| x4| y4| text
  ----------------------------|---|---|---|---|---|---|---|---|------
  img_calligraphy_70001_bg.jpg|72 |53 |96 |53 |96 |358|72 |358|黎沈昨骑托那缝丁聚侮篮海炭
  img_calligraphy_70001_bg.jpg|46 |53 |70 |53 |70 |394|46 |394|缩蝇躁劣趋拴局伦绸启杭吭惯蛋仅
  img_calligraphy_70001_bg.jpg|20 |53 |44 |53 |44 |174|20 |174|效射市关蝉
 
 创建好的数据集图片(绿线为标记的坐标,输入ctpn的图片是不带绿线的):
 ![Image text](https://github.com/hwwu/cptn-crnn/blob/master/cptn/data/demo/img_calligraphy_70001_bg.jpg)
 
 lable格式在 data/demo/img_calligraphy_70001_bg.txt中

### 训练
  
  1.首先下载vgg16的模型
  https://github.com/tensorflow/models/tree/1af55e018eebce03fb61bba9959a04672536107d/research/slim
  ,放到data目录下
  
  2.运行 python3 main/train.py
  
  可以直接下载我训练好的模型:https://drive.google.com/open?id=1RwZb1HLG0vum-5RHZdSfqtDD2in_sNRD
 
 
 ## crnn
 
 ### 环境搭建
 
    1.安装 pytorch和warp-ctc
    根据cuda版本选择pytorch的安装文件
    pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
    pip3 install torchvision
    git clone https://github.com/SeanNaren/warp-ctc.git
    cd warp-ctc;mkdir build; cd build;cmake ..;make
    cd ./warp-ctc/pytorch_binding;python setup.py install
    将 pytorch_binding 中生成的warpctc_pytorch文件夹copy到crnn下
 

