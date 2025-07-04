# -*- coding: utf-8 -*-
"""
dl_infer.py

主推理脚本，包含模型加载、图片预处理、推理、后处理及结果可视化流程。
适用于YOLOv10模型的推理与检测。
"""
#import logging
import json
import yaml
import cv2
import time
import torch

from utils.nne_util import *
#from utils import model_util
from utils.model_util import *
from constants import LOGGING_NAMESPACE
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops
from ultralytics.engine.results import Results

# 日志记录器初始化
logger = logging.getLogger(LOGGING_NAMESPACE)

# 检测类别颜色映射
class_colors = {
    0: (0, 0, 255),    # 红色
    1: (255, 0, 0),    # 蓝色
}
# 检测类别名称
names = ["fire", "smoke"]

def gpu_infer(input_data, engine, context):
    """
    使用GPU进行推理，支持多输入输出绑定。
    Args:
        input_data: 预处理后的输入数据
        engine: 推理引擎对象
        context: 推理上下文
    Returns:
        outputs_datas: 推理输出结果列表
    """
    num_bindings = engine.num_bindings
    logger.info("num_bindings: {}".format(num_bindings))
    with engine:
        batch_size = 1
        bindings = []
        # 遍历所有绑定，分配输入输出内存
        for index in range(num_bindings):
            binding_shape = engine.get_binding_shape(index)
            np_type = nne_to_np_type(engine.get_binding_dtype(index))
            batch_binding_size = batch_size * np_type(1).nbytes
            for s in binding_shape:
                batch_binding_size *= s
            batch_binding_shape = (batch_size,) + binding_shape
            if engine.binding_is_input(index):
                batch_input_datas = []
                batch_input_datas.append(input_data)
                batch_input_datas = np.array(batch_input_datas)
                batch_mem = cuda.mem_alloc(batch_binding_size)
                cuda.memcpy_htod(batch_mem, batch_input_datas)
            else:
                batch_mem = cuda.mem_alloc(batch_binding_size)
            bindings.append({
                "batch_mem": batch_mem,
                "batch_binding_size": batch_binding_size,
                "batch_binding_shape": batch_binding_shape,
                "np_type": np_type
            })
        # 构建输入绑定并执行推理
        binding_inputs = [binding['batch_mem'].as_buffer(binding['batch_binding_size']) for binding in bindings]
        context.execute(batch_size, binding_inputs)
        outputs_datas = []
        for index in range(num_bindings):
            if not engine.binding_is_input(index):
                output_datas = np.empty(bindings[index]['batch_binding_shape'], bindings[index]['np_type'])
                cuda.memcpy_dtoh(output_datas, bindings[index]['batch_mem'])
                for idx, output_data in enumerate(output_datas):
                    outputs_datas.append(np.reshape(output_data, engine.get_binding_shape(index)))
        return outputs_datas


class YOLOv10DetectionPredictor(DetectionPredictor):
    """
    YOLOv10 检测后处理类，继承自ultralytics的DetectionPredictor。
    """
    def postprocess(self, preds, img, orig_img):
        """
        对模型输出进行后处理，包括坐标变换、置信度筛选、类别筛选等。
        Args:
            preds: 原始模型输出
            img: 网络输入图片（张量）
            orig_img: 原始图片（ndarray）
        Returns:
            results: 检测结果列表，每项为单个目标的检测框、置信度、类别
        """
        if isinstance(preds, dict):
            preds = preds["one2one"]
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if preds.shape[-1] == 6:
            pass
        else:
            preds = preds.transpose(-1, -2)
            bboxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, preds.shape[-1] - 4)
            bboxes = ops.xywh2xyxy(bboxes)
            preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
        mask = preds[..., 4] > self.args.conf
        if self.args.classes is not None:
            mask = mask & (preds[..., 5:6] == torch.tensor(self.args.classes, device=preds.device).unsqueeze(0)).any(2)
        preds = [p[mask[idx]] for idx, p in enumerate(preds)]
        results = []
        for i, pred in enumerate(preds):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            for det in pred:
                results.append(det.tolist())
        return results


def model_init(model_path, config):
    """
    初始化模型，构建推理引擎与上下文。
    Args:
        model_path: 模型文件路径
        config: 配置字典
    Returns:
        engine: 推理引擎
        context: 推理上下文
    """
    with nne.Builder() as builder, nne.Parser() as parser:
        network = builder.create_network()
        weight_share_str = config['weight_share']
        weight_share = get_weight_share(weight_share_str)
        weight_mode = weight_share['weight_mode']
        cluster_cfg = weight_share['cluster_cfg']
        builder.config.max_batch_size = 1
        builder.config.ws_mode = weight_mode
        if config['build_set_flag'] == 'spm_alloc':
            builder.set_flag(nne.BuilderFlag.spm_alloc)
        logger.info("weight_share_str: {}".format(weight_share_str))
        parser.parse(model_path, network)
        engine = builder.build_engine(network)
        context = engine.create_execution_context(cluster_cfg)
        return engine, context
    

def preprocess_image(image_path, INPUT_W=640, INPUT_H=640):
    """
    图像预处理：读取、缩放、填充、归一化、格式变换。
    Args:
        image_path: 图片路径
        INPUT_W: 网络输入宽
        INPUT_H: 网络输入高
    Returns:
        image: 预处理后图片 (NCHW)
        image_raw: 原始图片
        h: 原图高
        w: 原图宽
    """
    image_raw = cv2.imread(image_path)         # 1. 读入图片
    h, w, c = image_raw.shape                  # 2. 记录图片大小
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)  # 3. BGR转RGB
    # 计算缩放比例
    r_w = INPUT_W / w
    r_h = INPUT_H / h
    if r_h > r_w:
        tw = INPUT_W
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((INPUT_H - th) / 2)
        ty2 = INPUT_H - th - ty1
    else:
        tw = int(r_h * w)
        th = INPUT_H
        tx1 = int((INPUT_W - tw) / 2)
        tx2 = INPUT_W - tw - tx1
        ty1 = ty2 = 0
    # 缩放并填充
    image = cv2.resize(image, (tw, th),interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114)
    )
    image = image.astype(np.float32)
    image /= 255.0
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = np.ascontiguousarray(image)
    return image, image_raw, h, w


def is_valid_img_file(img_name):
    """
    判断文件名是否为有效图片格式。
    """
    return img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png')


def get_total_img_paths_in_dir(dir_path):
    """
    获取目录下所有图片文件的完整路径。
    """
    assert os.path.exists(dir_path)
    img_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if is_valid_img_file(file):
                img_path = os.path.join(root, file)
                img_paths.append(img_path)
    return img_paths


def load_local_config(config_file_path):
    """
    加载本地配置文件（支持json/yaml）。
    """
    if config_file_path.endswith('json'):
        with open(config_file_path, "r") as f:
            _ = json.load(f)
            return _
    if config_file_path.endswith('yaml'):
        with open(config_file_path, "r") as f:
            _ = yaml.load(f, Loader=yaml.FullLoader)
            return _

#========================模型转换命令============================
# python3 -m dl convert /run/media/root/DATA/GxIn_ws/denglin/dl_infer_demo/models/yolov5s.onnx --input-shapes images:[1,3,640,640]
#==============================================================

if __name__ == '__main__':
    # 主流程入口，加载配置、初始化模型、遍历图片推理并保存结果
    config_path = "/run/media/root/DATA/GxIn_ws/denglin_infer/configs/config.yaml"
    config = load_local_config(config_path)
    gpu_model_path = config["model_path"]
    compare_config = config["compare_templates"][config["do_compares"]]
    img_dir = '/run/media/root/DATA/GxIn_ws/denglin_infer/data/images'

    # 初始化推理引擎和后处理器
    engine, context = model_init(gpu_model_path, compare_config)
    postprocess_yolov10 = YOLOv10DetectionPredictor()

    for img_path in get_total_img_paths_in_dir(img_dir):
        # 计时开始
        st = time.time()
        filename = os.path.basename(img_path)
        image, image_raw, h, w = preprocess_image(img_path)
        _preds = gpu_infer(image, engine, context)
        preds = torch.tensor(_preds)
        preds = postprocess_yolov10.postprocess(preds, torch.tensor(image), image_raw)
        # 绘制检测框与标签
        for obj in preds:
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            confidence = obj[4]
            label = int(obj[5])
            color = class_colors.get(label, (0, 255, 255))
            cv2.rectangle(image_raw, (left, top), (right, bottom), color=color, thickness=2, lineType=cv2.LINE_AA)
            caption = f"{names[label]} {confidence:.2f}"
            w, h = cv2.getTextSize(caption, 0, 1, 1)[0]
            cv2.rectangle(image_raw, (left - 3, top - 33), (left + w + 10, top), color, -1)
            cv2.putText(image_raw, caption, (left, top - 5), 0, 1, (0, 0, 0), 1, 16)
        # 保存检测结果图片
        if not os.path.exists("./detect_res"):
            os.makedirs("./detect_res")
        cv2.imwrite("./detect_res/" + filename, image_raw)
        # 计时结束并输出耗时
        end = time.time()
        time_taken = (end - st) * 1000
        print(f"时间消耗：{time_taken}ms")
