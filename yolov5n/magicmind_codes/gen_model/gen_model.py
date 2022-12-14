import argparse
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
from magicmind.python.runtime import DataType
from adapter_model import append_yolov5_detect
from calibrator import CalibData

def pytorch_parser(args):
    # 创建MagicMind parser
    parser = Parser(mm.ModelKind.kPytorch)
    # 设置网络输入数据类型
    parser.set_model_param("pytorch-input-dtypes", [DataType.FLOAT32])
    # 创建一个空的网络实例
    network = mm.Network()
    # 使用parser将Caffe模型文件转换为MagicMind Network实例。
    assert parser.parse(network, args.pt_model).ok()
    # 设置网络输入数据形状
    input_dims = mm.Dims((args.batch_size, 3, args.input_height, args.input_width))
    assert network.get_input(0).set_dimension(input_dims).ok()
    network = append_yolov5_detect(network, args.conf_thres, args.iou_thres, args.max_det)
    return network

def generate_model_config(args):
    config = mm.BuilderConfig()
     # 指定硬件平台
    assert config.parse_from_string('{"archs":[{"mtp_372": [4,6,8]}]}').ok()
    # INT64 转 INT32
    assert config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}').ok()
    assert config.parse_from_string('{"opt_config":{"conv_scale_fold": true}}').ok()
    # 输入数据摆放顺序
    # PyTorch模型输入数据顺序为NCHW，如下代码转为NHWC输入顺序。
    # 输入顺序的改变需要同步到推理过程中的网络预处理实现，保证预处理结果的输入顺序与网络输入数据顺序一致。
    # 以下JSON字符串中的0代表改变的是网络第一个输入的数据摆放顺序。1则代表第二个输入，以此类推。
    assert config.parse_from_string('{"convert_input_layout": { "0": {"src": "NCHW", "dst": "NHWC"}}}').ok()
    assert config.parse_from_string('{"cross_compile_toolchain_path": "/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/"}').ok()
    # 模型输入输出规模可变功能
    if args.shape_mutable == "true":
        assert config.parse_from_string('{"graph_shape_mutable": true}').ok()
        assert config.parse_from_string('{"dim_range": {"0": {"min": [1, 3, 640, 640], "max": [64, 3, 640, 640]}}}').ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable": false}').ok()
    # 精度模式
    assert config.parse_from_string('{"precision_config":{"precision_mode":"%s"}}' % args.quant_mode).ok()
    # 量化算法，支持对称量化（symmetric)和非对称量化（asymmetric）。当量化统计算法设置为EQNM_ALOGORITHM时，仅适用于对称量化。
    assert config.parse_from_string('{"precision_config": {"activation_quant_algo": "symmetric"}}').ok()
    # 设置量化粒度，支持按tensor量化（per_tensor）和按通道量化（per_axis）两种。
    assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_tensor"}}').ok()
    # 将网络预处理的标准化过程（减均值、除标准差: (input - mean) / std）放到模型中处理。将网络输入的标准化配置到模型中后，推理过程的预处理代码中就不再需要执行标准化相关计算。
    # 以下JSON字符串中键值0代表处理的是网络的第一个输入。mean数组中的值代表输入各通道的均值。var数组中的值代表输入各通道的方差即std²。
    # 本例中使用的yolov5模型mean为0，std为255（即var=65025)。
    assert config.parse_from_string('{"insert_bn_before_firstnode": {"0": {"mean": [0, 0, 0], "var": [65025, 65025, 65025]}}}').ok()
    return config

def calibrate(args, network : mm.Network, config : mm.BuilderConfig):
    # 创建量化工具并设置量化统计算法
    calib_data = CalibData(shape = mm.Dims([args.batch_size, 3, args.input_height, args.input_width]), max_samples = 10, img_dir = args.image_dir)
    calibrator = mm.Calibrator([calib_data])
    assert calibrator is not None
    # 设置量化统计算法，支持线性统计算法（LINEAR_ALGORITHM）及加强的最小化量化噪声算法（EQM_ALGORITHM）。
    assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
    # 打开设备
    with mm.System() as mm_sys:
        dev_count = mm_sys.device_count()
        print("Device count: ", dev_count)
        if args.device_id >= dev_count:
            print("Invalid device set!")
            # 打开MLU设备
            dev = mm.Device()
            dev.id = args.device_id
            assert dev.active().ok()
    # 进行量化
    assert calibrator.calibrate(network, config).ok()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", dest = 'device_id', default = 0, type = int, help = 'mlu device id, used for calibration')
    parser.add_argument("--pt_model", "--pt_model", type=str, default="../data/models/yolov5m_traced.pt", help="modified yolov5s pt")
    parser.add_argument("--output_model", "--output_model", type=str, default="../data/models/yolov5_pytorch_model", help="save mm model to this path")
    parser.add_argument("--image_dir", "--image_dir",  type=str, default="../../datasets/coco/val2017", help="coco2017 datasets")
    parser.add_argument("--quant_mode", "--quant_mode", type=str, default="qint8_mixed_float16", help="qint8_mixed_float16, force_float32, force_float16")
    parser.add_argument("--shape_mutable", "--shape_mutable", type=str, default="false", help="whether the mm model is dynamic or static or not")
    parser.add_argument("--batch_size", "--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument('--input_width', dest = 'input_width', default = 640, type = int, help = 'model input width')
    parser.add_argument('--input_height', dest = 'input_height', default = 640, type = int, help = 'model input height')
    parser.add_argument("--conf_thres", "--conf_thres", type=float, default=0.001, help="confidence_thresh")
    parser.add_argument("--iou_thres", "--iou_thres", type=float, default=0.65, help="nms_thresh")
    parser.add_argument("--max_det", "--max_det", type=int, default=1000, help="limit_detections")
    args = parser.parse_args()
    
    supported_quant_mode = ['qint8_mixed_float16', 'qint8_mixed_float32', 'force_float16', 'force_float32']
    if args.quant_mode not in supported_quant_mode:
        print('quant_mode [' + args.quant_mode + ']', 'not supported')
        exit()

    network = pytorch_parser(args)
    config = generate_model_config(args)

    if args.quant_mode.find('qint') != -1:
        print('do calibrate...')
        calibrate(args, network, config)
    print('build model...')
    # 生成模型
    builder = mm.Builder()
    model = builder.build_model("yolov5_pytorch_model", network, config)
    assert model != None
    # 将模型序列化为离线文件
    model.serialize_to_file(args.output_model)
    print("Generate model done, model save to %s" % args.output_model)

if __name__ == "__main__":
    main()
