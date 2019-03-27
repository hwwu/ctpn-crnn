# cptn-crnn

## cptn

1.环境搭建(tqdm,opencv-python,Shapely,matplotlib,numpy,tensorflow-gpu or tensorflow,Cython,ipython 自行用pip3安装)
 cd cptn/utils/bbox
 sh make.sh

2.创建数据集
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

 
