# 训练一个自己的YOLO-tiny模型

## 0x01 YOLO是什么
[YOLO](https://pjreddie.com/darknet/yolo/)是一个实时目标检测系统，通俗点说就是在输入数据(图片或者视频)中查找特定的目标。举个例子，如果让一个专门识别龙的YOLO模型观看《权力的游戏》，在理想情况下，一旦画面中出现了龙，YOLO系统就会激动地用框框标记出画面中的龙。

### 为什么是YOLO-tiny
大概是贫穷限制了我的运算速度吧，YOLO-tiny用精确度换来了速度快和性能要求低的优点，适合练手和学习或者像我一样玩一玩的用户。

## 0x02 事前准备的准备
首先你可能需要一台电脑吧（可能有些用户会使用树莓派甚至是装了linux模拟器的安卓用户来虐待自己吧）。我的CPU是不知道应该叫啥四代ES版本的i7,显卡是上古时代的GTX860M，内存只有8G，名副其实的低配置用户了。顺便一提，这台电脑一天会绿屏N次，我会从现在开始记录绿屏次数，直到这篇文章写完，再顺带一提，这是二奶机，因为大奶机用的AMD矿卡，所以不能使用GPU来加速TensorFlow训练。再再顺带一提，旁边的室友虽然是1070Ti，但是他需要用他的电脑玩欢乐斗地主，所以不能帮我跑训练模型。
回到主线上来，你还得具有一些常识...会使用搜索引擎的那种常识。

## 0x03 事前准备
这里假设你已经安装好了Python，并且具备了一些"常识"后，你就可以继续往下看了。
为了方便演示，我使用了Python的虚拟环境，并且安装了如下库

<img src="https://github.com/chaosgoo/imgtag_tutorial/blob/master/images/images_11.png?raw=true"/>

* 下载 https://github.com/qqwweee/keras-yolo3 页面中的文件，你可以使用git clone或者直接点击下载，然后解压。
* 下载 yolo-tiny的weights文件 https://pjreddie.com/media/files/yolov3-tiny.weights, 并放到上一步解压后的文件夹中
进行到这里，我们接下来需要用到的文件已经准备了一半了，另外一半就是跑模型需要用到的数据。

## 0x04 配置相关文件
首先，我们需要对yolov3-tiny.cfg进行编辑，
在这个文件里，我们需要关注[yolo]和[yolo]前的一个[convolutional]
首先将所有[yolo]里面classes修改为1
得到classes=1
classes的含义是有多少种需要被识别的物体，这里我只训练yolo识别一种物体，所以设置为1

对所有[yolo]的前一个[convolutional]中的filters进行修改
其取值为filters = 3 * ( classes + 5 ),由于上一步中classes=1所以这里filters取18
到这里，yolov3-tiny.cfg就修改完毕了
然后是修改model_data中的coco_classes.txt和voc_classes.txt，将待检测物体的标签填写进去，每种标签占一行。因为我只有一种待识别物体，所以这两个文件中都只有一个单词
进入到yolo所在目录，运行
````bash
python convert.py -w yolov3-tiny.cfg yolov3-tiny.weights model_data/tiny_yolo_weights.h5
````
转换完成后可以看见如图所示内容

<img src="https://github.com/chaosgoo/imgtag_tutorial/blob/master/images/images_01.png?raw=true"/>

## 0x05 当然是制作数据啦,DIO
怎么制作数据呢...我写了ImgTag去标注数据，并且在训练过程中使用ImgTag产生的数据

## 0x06 训练模型
在训练自己的模型之前，我们还需要编辑一下train.py中的内容
* anchors_path = 'model_data/tiny_yolo_anchors.txt' 指定anchors为tiny-yolo版本
* 因为我已经明确知道我训练的是yolo-tiny模型，所以在27行，修改为is_tiny_version = True
此时就可以执行
````bash
python train.py
````
漫长的等待之后，模型就跑完了.跑完以后会显示类似内容

<img src="https://github.com/chaosgoo/imgtag_tutorial/blob/master/images/images_02.png?raw=true"/>

## 0x07 使用模型
使用模型，就是指定yolo在检测目标时候，使用我们刚刚（可能并不是刚刚）产生的权重文件。
进入到yolo的目录，编辑yolo.py
其中需要重点关注的内容就是
````python
 _defaults = {
    "model_path": 'model_data/yolo.h5', # 指定使用的模型
    "anchors_path": 'model_data/yolo_anchors.txt',
    "classes_path": 'model_data/coco_classes.txt',
    "score" : 0.3, # 当评估出的得分大于0.3时候，就标记出来
    "iou" : 0.45,
    "model_image_size" : (416, 416),
    "gpu_num" : 1,
}
````
我们需要修改以下model_path的值，因为..某些特殊的原因，我在运行的时候指定模型并不能正常进行，所以我使用这种方法。
model_path那一行修改为
````python
"model_path": 'logs/000/trained_weights_final.h5',
````
我一共标注了200张图片，相对而言图片数量可能还不够，但是已经可以看到目前模型已经可以正常识别部分浣熊了。比如银河护卫队中的火箭浣熊哦

<img src="https://github.com/chaosgoo/imgtag_tutorial/blob/master/images/raccoon.gif?raw=true"/>