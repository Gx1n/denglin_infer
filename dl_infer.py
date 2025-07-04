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

logger = logging.getLogger(LOGGING_NAMESPACE)

class_colors = {
    0: (0, 0, 255),    # 红色
    1: (255, 0, 0),    # 蓝色
}
names = ["fire", "smoke"]


def cpu_infer(model_path, inputs_datas, outputs_name=None):
    model_suffix = extract_model_suffix(model_path)
    if model_suffix == '.onnx':
        return onnx_infer(model_path, inputs_datas, _outputs_name=outputs_name)

    raise AssertionError("model type have not support! model type is: {}".format(model_suffix))


def onnx_infer(model_path, inputs_datas, _outputs_name=None):
    if _outputs_name is None:
        _outputs_name = []
    import onnxruntime
    import onnx

    model = onnx.load(model_path)

    if _outputs_name and len(_outputs_name) != 0:
        model.graph.output.extend([onnx.ValueInfoProto(name=output) for output in _outputs_name])

    onnx_session = onnxruntime.InferenceSession(model.SerializeToString())
    inputs_name = []
    for node in onnx_session.get_inputs():
        inputs_name.append(node.name)

    if _outputs_name and len(_outputs_name) != 0:
        outputs_name = _outputs_name
    else:
        outputs_name = [node.name for node in onnx_session.get_outputs()]

    outputs_datas = []

    for idx, inputs_data in enumerate(inputs_datas):
        outputs_data = onnx_session.run(outputs_name, inputs_data)
        output_dict = {}
        for i, output_name in enumerate(outputs_name):
            output_dict[output_name] = outputs_data[i]
        outputs_datas.append(output_dict)

    return outputs_datas




def gpu_infer(input_data, engine, context):
        num_bindings = engine.num_bindings
        logger.info("num_bindings: {}".format(num_bindings))
        with engine:
            batch_size = 1
            bindings = []

            for index in range(num_bindings):
                # binding_name = engine.get_binding_name(index)
                binding_shape = engine.get_binding_shape(index)

                np_type = nne_to_np_type(engine.get_binding_dtype(index))

                batch_binding_size = batch_size * np_type(1).nbytes
                for s in binding_shape:
                    batch_binding_size *= s

                batch_binding_shape = (batch_size,) + binding_shape

                if engine.binding_is_input(index):
                    batch_input_datas = []
                    # input_data = inputs_data[binding_name]
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

            binding_inputs = [binding['batch_mem'].as_buffer(binding['batch_binding_size']) for binding in bindings]
            context.execute(batch_size, binding_inputs)

            # outputs_datas = [{} for _ in range(batch_size)]
            outputs_datas = []
            for index in range(num_bindings):
                if not engine.binding_is_input(index):
                    # binding_name = engine.get_binding_name(index)
                    output_datas = np.empty(bindings[index]['batch_binding_shape'], bindings[index]['np_type'])
                    cuda.memcpy_dtoh(output_datas, bindings[index]['batch_mem'])
                    # print(bindings[index]['batch_binding_shape'])
                    # print(bindings[index]['batch_mem'])
                    for idx, output_data in enumerate(output_datas):
                        # outputs_datas[idx][binding_name] = np.reshape(output_data, engine.get_binding_shape(index))
                        outputs_datas.append(np.reshape(output_data, engine.get_binding_shape(index)))
            return outputs_datas


class YOLOv10DetectionPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_img):
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

        # if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        #     orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            #orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            # img_path = self.batch[0][i]
            for det in pred:
                results.append(det.tolist())
            #results.append(Results(orig_img, names=self.model.names, boxes=pred))
        return results


def model_init(model_path, config):
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
    description: Read an image from image path, convert it to RGB,
                 resize and pad it to target size, normalize to [0,1],
                 transform to NCHW format.
    param:
        image_path: str, image path
    return:
        image:  the processed image
        image_raw: the original image
        h: original height
        w: original width
    """
    image_raw = cv2.imread(image_path)         # 1.opencv读入图片
    h, w, c = image_raw.shape                  # 2.记录图片大小
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)  # 3. BGR2RGB
    # Calculate widht and height and paddings
    r_w = INPUT_W / w  # INPUT_W=INPUT_H=640  # 4.计算宽高缩放的倍数 r_w,r_h
    r_h = INPUT_H / h
    if r_h > r_w:       # 5.如果原图的高小于宽(长边），则长边缩放到640，短边按长边缩放比例缩放
        tw = INPUT_W
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((INPUT_H - th) / 2)  # ty1=（640-短边缩放的长度）/2 ，这部分是YOLOv5为加速推断而做的一个图像缩放算法
        ty2 = INPUT_H - th - ty1       # ty2=640-短边缩放的长度-ty1
    else:
        tw = int(r_h * w)
        th = INPUT_H
        tx1 = int((INPUT_W - tw) / 2)
        tx2 = INPUT_W - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th),interpolation=cv2.INTER_LINEAR)  # 6.图像resize,按照cv2.INTER_LINEAR方法
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        # image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114)

    )  # image:图像， ty1, ty2.tx1,tx2: 相应方向上的边框宽度，添加的边界框像素值为常数，value填充的常数值
    image = image.astype(np.float32)   # 7.unit8-->float
    # Normalize to [0,1]
    image /= 255.0    # 8. 逐像素点除255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])   # 9. HWC2CHW
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)    # 10.CWH2NCHW
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)  # 11.ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
    return image, image_raw, h, w  # 处理后的图像，原图， 原图的h,w

def is_valid_img_file(img_name):
    return img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png')

def get_total_img_paths_in_dir(dir_path):
    assert os.path.exists(dir_path)
    img_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if is_valid_img_file(file):
                img_path = os.path.join(root, file)
                img_paths.append(img_path)
    return img_paths

def load_local_config(config_file_path):
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
    config_path = "/run/media/root/DATA/GxIn_ws/denglin_infer/configs/config.yaml"
    config = load_local_config(config_path)
    gpu_model_path = config["model_path"]
    compare_config = config["compare_templates"][config["do_compares"]]
    img_dir = '/run/media/root/DATA/GxIn_ws/denglin_infer/data/images'

    # 初始化
    engine, context = model_init(gpu_model_path, compare_config)
    postprocess_yolov10 = YOLOv10DetectionPredictor()

    for img_path in get_total_img_paths_in_dir(img_dir):
        # 计时
        st = time.time()

        filename = os.path.basename(img_path)

        image, image_raw, h, w = preprocess_image(img_path)
        _preds = gpu_infer(image, engine, context)
        preds = torch.tensor(_preds)
        preds = postprocess_yolov10.postprocess(preds, torch.tensor(image), image_raw)

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

        if not os.path.exists("./detect_res"):
            os.makedirs("./detect_res")
        cv2.imwrite("./detect_res/" + filename, image_raw)


        end = time.time()
        time_taken = (end - st) * 1000
        print(f"时间消耗：{time_taken}ms")
