import time
import cv2 as cv
import numpy as np
import onnxruntime
from PIL import Image, ImageDraw, ImageFont


CLASSES=['人', '自行车', '小车', '摩托车', '飞机', '大巴', '火车', '货车', '船', '红绿灯',
        '消防栓', '停车标志', '停车计时器', '长椅', '鸟', '猫', '狗', '马', '羊', '牛',
        '大象', '熊', '斑马', '长颈鹿', '背包', '雨伞', '手提包', '领带', '手提箱', '飞盘',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', '杯子', '叉子', '小刀', 'spoon', '碗', '香蕉', '苹果',
        '三明治', 'orange', 'broccoli', 'carrot', '热狗', 'pizza', 'donut', '蛋糕', 'chair', 'couch',
        '盆栽植物', '床', 'dining table', 'toilet', 'tv', 'laptop', '鼠标', '遥控器', '键盘', '电话',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        '吹风机', '牙刷'] #coco80类别
TATGET = ['人', '自行车', '小车', '摩托车', '大巴', '火车', '货车', '电话']

class YOLOV5():    #yolov5 onnx推理
    def __init__(self,onnxpath):
        self.onnx_session=onnxruntime.InferenceSession(onnxpath)
        self.input_name=self.get_input_name()
        self.output_name=self.get_output_name()
    def get_input_name(self):
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
    def get_output_name(self):
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
    def get_input_feed(self,img_tensor):
        input_feed={}
        for name in self.input_name:
            input_feed[name]=img_tensor
        return input_feed

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] / hw_scale)
                img = cv.resize(srcimg, (neww, newh), interpolation=cv.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv.copyMakeBorder(img, 0, 0, left, self.input_shape[1] - neww - left, cv.BORDER_CONSTANT,
                                            value=0)  # add border
            else:
                newh, neww = int(self.input_shape[0] * hw_scale), self.input_shape[1]
                img = cv.resize(srcimg, (neww, newh), interpolation=cv.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv.copyMakeBorder(img, top, self.input_shape[0] - newh - top, 0, 0, cv.BORDER_CONSTANT, value=0)
        else:
            img = cv.resize(srcimg, self.input_shape, interpolation=cv.INTER_AREA)
        return img, newh, neww, top, left

    
    def inference(self,img):
        # img=cv.imread(img_path)   #读取图片

        or_img=cv.resize(img,(640,640))
        # or_img=cv.resize(img,(640,640))
        # print(f'org_img type is =  {type(or_img)}')
        img=or_img[:,:,::-1].transpose(2,0,1)  #BGR2RGB和HWC2CHW
        img=img.astype(dtype=np.float32)
        img/=255.0
        img=np.expand_dims(img,axis=0)
        input_feed=self.get_input_feed(img)
        pred=self.onnx_session.run(None,input_feed)[0]
        return pred,or_img

def pynms(dets, thresh): #非极大抑制
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1] #置信度从大到小排序（下标）

    while index.size > 0:
        i = index[0]
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # 计算相交面积
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # 当两个框不想交时x22 - x11或y22 - y11 为负数，
                                           # 两框不相交时把相交面积置0
        h = np.maximum(0, y22 - y11 + 1)  #

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)#计算IOU

        idx = np.where(ious <= thresh)[0]  #IOU小于thresh的框保留下来
        index = index[idx + 1]  # 下标以1开始
        # print(index)

    return keep


def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def filter_box(org_box,conf_thres,iou_thres): #过滤掉无用的框
    org_box=np.squeeze(org_box) #删除为1的维度
    conf = org_box[..., 4] > conf_thres #删除置信度小于conf_thres的BOX
    # print(conf)
    box = org_box[conf == True]
    cls_cinf = box[..., 5:]
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))
    all_cls = list(set(cls))     #删除重复的类别
    output = []
    for i in range(len(all_cls)):
        curr_cls = all_cls[i]
        curr_cls_box = []
        curr_out_box = []
        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls #将第6列元素替换为类别下标
                curr_cls_box.append(box[j][:6])   #当前类别的BOX
        curr_cls_box = np.array(curr_cls_box)
        curr_cls_box = xywh2xyxy(curr_cls_box)
        curr_out_box = pynms(curr_cls_box,iou_thres) #经过非极大抑制后输出的BOX下标
        for k in curr_out_box:
            output.append(curr_cls_box[k])  #利用下标取出非极大抑制后的BOX
    output = np.array(output)
    return output

def draw(image,box_data):  #画图
    print(box_data)
    boxes=box_data[...,:4].astype(np.int32) #取整方便画框
    scores=box_data[...,4]
    classes=box_data[...,5].astype(np.int32) #下标取整

    # nums_dict = {'person': 0, 'car':0,  'truck':0, 'motorbike':0}
    nums_dict = dict({})
    for i in range(len(CLASSES)):
          nums_dict[CLASSES[i]] = 0
    # print(nums_dict)
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        cv.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        if CLASSES[cl] in TATGET:
            if (isinstance(image, np.ndarray)):  #判断是否OpenCV图片类型
                img = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img)    # 创建绘制图像
                txt = '{0} {1:.2f}'.format(CLASSES[cl], score)
                fontstype = ImageFont.truetype("myfont.ttf", 20, encoding="utf-8")
                # print('class: {}, score: {}'.format(CLASSES[cl], score))
                # print(f' cl ---------- {cl}')
                draw.text((top, left-25), txt, (0, 255, 0), font=fontstype)  # 绘制文本
                image = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

                if cl == 0:
                    nums_dict['人'] += 1
                    # print((nums_dict['人']))
                if cl == 7:
                    nums_dict['货车'] += 1
                    # print((nums_dict['货车']))
                if cl == 2:
                    nums_dict['小车'] += 1
                    # print((nums_dict['小车']))
                # txt = r'{0} {1:.2f}'.format(CLASSES[cl], score)
                # print(f'(top left)  = {(top, left)}')
                # print(txt)
                # image = cvImgAddText(image, txt, left-25, top)
                # img = cvImgAddText(image, txt, t, l)
        # else:
        #     # print(CLASSES[cl])
        #     cv.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
        #                 (top, left ),
        #                 cv.FONT_HERSHEY_SIMPLEX,
        #                 0.6, (0, 0, 255), 2)
    return image, nums_dict
    # cv.imshow('res', img)
    # return cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
    # return image

def cvImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./myfont.ttf", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)





if __name__=="__main__":

    url_web = r'rtmp://xyrtmp.ys7.com:1935/v3/openlive/J17645702_1_1?expire=1691889301&id=480303600422105088&t=328fccd0260e758fe0e4be7e0068109d54d3977696105a8f0da2adae8f9b69ab&ev=100'
    # url_web = f'images/test.mp4'
    onnx_path = r'E:\source\code\vision\models\weights\yolov5\yolov5s.onnx'
    # onnx_path = r'E:\source\code\vision\weights\csd\yolov5s.onnx'
    model=YOLOV5(onnx_path)

    my_sizes = (800, 600)
    # pos1 = np.float32([[535, 94], [85, 471], [1224, 164], [1140, 847]])
    pos1 = np.float32([[535, 94], [85, 471], [1224, 94], [1140, 847]])
    pos2 = np.float32([[0, 0], [0, 880], [1480, 0], [1480, 880]])

    M = cv.getPerspectiveTransform(pos1, pos2)



    cap = cv.VideoCapture(url_web)
    cap_flag = cap.isOpened()
    print(f'cap is open = {cap_flag}')
    # frames = cap.get(cv.CAP_PROP_FRAME_COUNT)#获得视频文件的帧数
    # fps = cap.get(cv.CAP_PROP_FPS)#获得视频文件的帧率
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)#获得视频文件的帧宽
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)#获得视频文件的帧高
    while cap_flag:
        ret, frame = cap.read()
        # frame = frame[98:529, 877:1131]
        # frame = cv.warpPerspective(frame, M, (int(width), int(height)))
        # frame = cv.warpPerspective(frame, M, (1920, 1080))    # 透视变换
        # frame1 = frame
        # if not frame:
        #     continue
        # print(type(frame))
        if isinstance(frame, np.ndarray):
            if frame.size > 0:
                frame = frame[55:, 30:-350]
        else:
            continue

        if not ret:
            break
        # cv.putText(frame,"Press Q to save and quit", (10, 130), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # frame = cvImgAddText(frame, f'有：    人', 0, 100 , (255,0, 0), textSize=20)
        # frame = cvImgAddText(frame, f'有：    车', 0, 120, (255, 0, 0), textSize=20)
        # im = cvImgAddText(im, f'有：    小车', 0, 140, (255, 0, 0))
        # 缩放图像(宽、高)
        # frame = cv.resize(frame, my_sizes)
        output, img = model.inference(frame)

        outbox = filter_box(output, 0.5, 0.5)
        print(" = " * 15)
        print(outbox.shape)
        print(outbox)
        # if outbox == []:
        # if outbox is not None:
        if outbox.size != 0:
            # print(outbox)
            print(' - ' * 10)
            frame, nums_dict = draw(img, outbox)

            # im = cvImgAddText(im, f'{nums_dict["人"]}', 0, 100 , (255,0, 0))
            # im = cvImgAddText(im, f'{nums_dict["小车"]}', 0, 120, (255, 0, 0))
            # im = cvImgAddText(im, f'{nums_dict["货车"]}', 0, 140, (255, 0, 0))
        frame = cvImgAddText(frame, f'有：{nums_dict["人"]} 人', 0, 100 , (255,0, 0))
        frame = cvImgAddText(frame, f'有：{nums_dict["货车"]} 部货车', 0, 120, (255, 0, 0))
        frame = cvImgAddText(frame, f'有：{nums_dict["小车"]} 小车', 0, 140, (255, 0, 0))
        # frame = cv.resize(frame,(1024, 840))
        cv.imshow('video', frame)
       
        # else:
        #     img = cv.resize(img,(1024, 840))
        #     cv.imshow('video', img)
        
        # 每帧数据延时 1ms, 延时为0, 读取的是静态帧
        k = cv.waitKey(5)
        if k == ord('s'):
            tt = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())

            # cv2.imwrite("test.jpg", frame)
            cv.imwrite("results/{}.png".format(tt), frame)
        
        if k == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()   







