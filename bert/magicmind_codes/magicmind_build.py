# 全局变量代码块
import logging
import magicmind.python.runtime as mm
import os
import time
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV1Processor
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate
import warnings
import sys
warnings.filterwarnings("ignore")

# 模型单次推理批大小
BATCH_SIZE = 128

MAX_SEQ_LENGTH = 128

DATA_TYPE = sys.argv[1]

# PyTorch模型文件保存路径
PYTORCH_MODEL_PATH = "./models/pt_model/bert_base_cased_squad.pt"

# MagicMind离线模型存放路径
OFFLINE_MODEL_PATH = "./models/mm_model/bert_base_cased_squad_" + DATA_TYPE + ".mm"

# MLU设备ID。如：4卡机器，则可设置值为 0，1，2，3
DEV_ID = 0

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# 生成Pytorch模型文件
pt_model = AutoModelForQuestionAnswering.from_pretrained(
    "./weights",
    from_tf = False,
    config = None,
    cache_dir = None
)

pt_model.load_state_dict(
    torch.load("./weights/pytorch_model.bin",
    map_location = 'cpu')
)

pt_model.eval()

tokens = torch.randint(0, 1, (1, MAX_SEQ_LENGTH))
segments = torch.randint(0, 1, (1, MAX_SEQ_LENGTH))
mask = torch.randint(0, 1, (1, MAX_SEQ_LENGTH))

# 保存模型文件
torch.jit.save(torch.jit.trace(pt_model, (tokens, segments, mask)), PYTORCH_MODEL_PATH)

# pt_model  to mm_model
from magicmind.python.runtime.parser import Parser
# step1: 创建MagicMind PyTorch parser
mm_parser = Parser(mm.ModelKind.kPytorch)

# step2: 设置网络输入数据类型
mm_parser.set_model_param("pytorch-input-dtypes", [mm.DataType.INT32] * 3)

# step3: 创建一个空的网络实例
mm_network = mm.Network()

# step4: 输入PyTorch模型文件，转换MagicMind网络
status = mm_parser.parse(mm_network, PYTORCH_MODEL_PATH)
assert status.ok()

# 通过json字符串配置Builder参数
config = mm.BuilderConfig()
# INT64转INT32
assert config.parse_from_string("{\"opt_config\":{\"type64to32_conversion\": true}}").ok()

# 
assert config.parse_from_string("""{"archs": ["mtp_372"]}""").ok()

# 精度模式，BERT网络在当前MagicMind版本中float16精度模式最佳
if DATA_TYPE == "fp32":
  assert config.parse_from_string("{\"precision_config\":{\"precision_mode\": \"force_float32\"}}").ok()
elif DATA_TYPE == "fp16":
  assert config.parse_from_string("{\"precision_config\":{\"precision_mode\": \"force_float16\"}}").ok()
elif DATA_TYPE == "int8":
  assert config.parse_from_string("{\"precision_config\":{\"precision_mode\": \"qint8_mixed_float16\"}}").ok()
else:
  print("Invalid DATA_TYPE!!!!!")
  exit
  
# 生成可变形状的mm model
assert config.parse_from_string('{ \
    "archs": [{"mtp_372": [4]}],  \
    "graph_shape_mutable": true,  \
    "dim_range": {  \
      "0": {  \
        "min": [1, 1],  \
        "max": [%d, %d]  \
      },  \
      "1": {  \
        "min": [1, 1],  \
        "max": [%d, %d]  \
      },  \
      "2": {  \
        "min": [1, 1],  \
        "max": [%d, %d]  \
      }  \
    }}' % ((BATCH_SIZE, MAX_SEQ_LENGTH) * 3)).ok()

# 设置模型输入形状和数据类型
for i in range(mm_network.get_input_count()):
    mm_network.get_input(i).set_data_type(mm.DataType.INT32)
    mm_network.get_input(i).set_dimension(mm.Dims((BATCH_SIZE, MAX_SEQ_LENGTH)))

# 生成模型
mm_builder = mm.Builder()
mm_model = mm_builder.build_model("bert", mm_network, config)
assert mm_model is not None

# 调用Model.serialize_to_file接口完成将生成的模型保存为离线文件
assert mm_model.serialize_to_file(OFFLINE_MODEL_PATH).ok()
print("Build MM Model Success!")

