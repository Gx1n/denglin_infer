import logging

import os
import sys
import pycuda.driver as cuda
import dlnne as nne
import numpy as np
from tvm import relay
from tvm import topi
from tvm.driver import tvmc
# from utils import img_util
#
# from utils.constants import LOGGING_NAMESPACE
import img_util

from constants import LOGGING_NAMESPACE

logger = logging.getLogger(LOGGING_NAMESPACE)


def nne_to_np_type(_type):
    if _type == nne.DataType.FLOAT:
        np_type = np.float32
    elif _type == nne.DataType.HALF:
        np_type = np.float16
    elif _type == nne.DataType.UINT8:
        np_type = np.uint8
    elif _type == nne.DataType.UINT16:
        np_type = np.uint16
    elif _type == nne.DataType.UINT32:
        np_type = np.uint32
    elif _type == nne.DataType.UINT64:
        np_type = np.uint64
    elif _type == nne.DataType.INT8:
        np_type = np.int8
    elif _type == nne.DataType.INT16:
        np_type = np.int16
    elif _type == nne.DataType.INT32:
        np_type = np.int32
    elif _type == nne.DataType.INT64:
        np_type = np.int64
    else:
        raise AssertionError("Unknown nne data type")
    return np_type


weight_share_configs = {
    "0": {
        "weight_mode": nne.WeightShareMode.single,
        "cluster_cfg": nne.ClusterConfig.cluser0,
    },
    "1": {
        "weight_mode": nne.WeightShareMode.single,
        "cluster_cfg": nne.ClusterConfig.cluser1,
    },
    "2": {
        "weight_mode": nne.WeightShareMode.single,
        "cluster_cfg": nne.ClusterConfig.cluser2,
    },
    "3": {
        "weight_mode": nne.WeightShareMode.single,
        "cluster_cfg": nne.ClusterConfig.cluser3,
    },
    "01": {
        "weight_mode": nne.WeightShareMode.share2,
        "cluster_cfg": nne.ClusterConfig.cluser01,
    },
    "23": {
        "weight_mode": nne.WeightShareMode.share2,
        "cluster_cfg": nne.ClusterConfig.cluser23,
    },
    "0123": {
        "weight_mode": nne.WeightShareMode.share4,
        "cluster_cfg": nne.ClusterConfig.cluser0123,
    },
}


def get_cluster_count(gpu=0):
    return cuda.get_cluster_count(gpu)


def get_weight_share(weight_share="all"):
    if weight_share == 'all':
        cluster_count = get_cluster_count()
        if cluster_count == 1:
            return weight_share_configs["0"]
        if cluster_count == 2:
            return weight_share_configs["01"]
        if cluster_count == 4:
            return weight_share_configs["0123"]
        raise AssertionError("not match correct cluster from all, need fix this code!")

    if weight_share not in weight_share_configs:
        raise AssertionError("weight_share: %s is not a valid input" % weight_share)
    return weight_share_configs[weight_share]


def get_params_info(model_path, params=None):
    tvm_model = tvmc.frontends.load_model(model_path)
    mod, params = tvm_model if isinstance(tvm_model, tuple) else (tvm_model.mod, tvm_model.params)
    if params:
        mod['main'] = relay.build_module.bind_params_by_name(mod['main'], params)
    if params is not None:
        func = relay.build_module.bind_params_by_name(mod["main"], params)
    else:
        func = mod["main"]
    inputs_name = []
    inputs_shape = []
    inputs_np_data_type = []
    for v in func.params:
        inputs_name.append(v.name_hint)
        inputs_shape.append(topi.utils.get_const_tuple(v.type_annotation.shape))
        inputs_np_data_type.append(v.type_annotation.dtype)
    return inputs_np_data_type, inputs_name, inputs_shape, mod, params


def generate_input(config):
    if config['input_type'] == 'random':
        random_input_config = config['input_types']['random']
        return generate_random_input(
            config['model_path'],
            random_input_config['batch_size'],
            random_input_config['input_min'],
            random_input_config['input_max'],
            random_input_config['seed'] if 'seed' in random_input_config and random_input_config['seed'] else None
        )
    if config['input_type'] == 'npy':
        npy_input_config = config['input_types']['npy']
        return generate_npy_input(npy_input_config['npy_inputs_data_dir'])
    if config['input_type'] == 'img':
        img_input_config = config['input_types']['img']
        if 'img_preprocess_holder_dir' in img_input_config and img_input_config['img_preprocess_holder_dir']:
            sys.path.append(img_input_config['img_preprocess_holder_dir'])
        try:
            import img_preprocess_holder
            pre_process_func = img_preprocess_holder.div_img_preprocess
        except ImportError:
            pre_process_func = img_util.load_and_resize_img

        logger.info("img_pre_process_func: {}".format(pre_process_func))

        return generate_input_by_img(
            config['model_path'],
            img_input_config['dir'],
            img_input_config['count'],
            pre_process_func
        )
    raise AssertionError("no matched correct inout_type: %s" % config['input_type'])


def generate_input_by_img(model_path, image_dir, count=-1, pre_process_func=None):
    inputs_data_type, inputs_name, inputs_shape, mod, params = get_params_info(model_path)

    img_input_name = ''
    data_np_type = None
    width = 0
    height = 0
    # find like img input by shape
    for idx, input_shape in enumerate(inputs_shape):
        if len(input_shape) >= 3:
            data_np_type = inputs_data_type[idx]
            img_input_name = inputs_name[idx]
            width = inputs_shape[idx][3]
            height = inputs_shape[idx][2]
            break

    if not img_input_name:
        raise AssertionError("not match correct img input, please check the code!")

    inputs_datas = img_util.load_img_data(img_input_name, image_dir, data_np_type,
                                          width=width,
                                          height=height,
                                          count=count,
                                          pre_process_func=pre_process_func)
    return inputs_datas


def generate_random_input(model_path, batch, input_min=0, input_max=1.0, seed=None):
    inputs_data_type, inputs_name, inputs_shape, mod, params = get_params_info(model_path)

    inputs_datas = []
    for idx in range(batch):
        input_dict = {}
        for i, input_name in enumerate(inputs_name):
            logger.info("Input {}: name = {}, shape = {}".format(i, input_name, inputs_shape[i]))
            if seed:
                np.random.seed(seed)
            input_data = np.random.random(inputs_shape[i])
            input_data = (input_data * (input_max - input_min)) - (input_max - input_min) / 2
            input_data = input_data.astype(inputs_data_type[i])
            input_dict[input_name] = input_data
        inputs_datas.append(input_dict)

    return inputs_datas


def generate_npy_input(npy_data_dir):
    inputs_datas = []
    inputs_data = {}
    for root, dirs, files in os.walk(npy_data_dir):
        for file in files:
            if not file.endswith(".npy"):
                continue
            input_name, ext = os.path.splitext(file)
            load_np_arr = np.load(os.path.join(root, file))
            inputs_data[input_name] = load_np_arr
    inputs_datas.append(inputs_data)
    return inputs_datas


def get_model_outputs(model_path):
    inputs_data_type, inputs_name, inputs_shape, mod, params = get_params_info(model_path)
    return get_rlym_outputs(mod)


def get_rlym_outputs(mod):
    outputs_name = []
    if mod["main"].attrs:
        if "outputs" in mod["main"].attrs and mod["main"].attrs["outputs"]:
            outputs_name = list(mod["main"].attrs["outputs"])
            return outputs_name
    ret_type = relay.transform.InferType()(mod)["main"].ret_type
