# 全局变量代码块
import logging
import magicmind.python.runtime as mm
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
import numpy
warnings.filterwarnings("ignore")

# 模型单次推理批大小
BATCH_SIZE = 16

MAX_SEQ_LENGTH = 128

DATA_TYPE = sys.argv[1]

# MagicMind离线模型存放路径
OFFLINE_MODEL_PATH = "./models/mm_model/bert_base_cased_squad_" + DATA_TYPE + ".mm"

# MLU设备ID。如：4卡机器，则可设置值为 0，1，2，3
DEV_ID = 0

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", do_lower_case = False)
squad_processor = SquadV1Processor()
examples = squad_processor.get_dev_examples("", filename="data/dev-v1.1.json")
features, dataset = squad_convert_examples_to_features(
    examples = examples,
    tokenizer = tokenizer,
    max_seq_length = MAX_SEQ_LENGTH,
    doc_stride = 128,
    max_query_length = 64,
    is_training = False,
    return_dataset = "pt",
    threads = 4)

eval_sampler = SequentialSampler(dataset)
eval_dataloader = DataLoader(dataset, sampler = eval_sampler, batch_size = BATCH_SIZE, drop_last = False)
print("Num examples = ", len(dataset))
print("Batch size = ", BATCH_SIZE)
print("Iterations = ", len(eval_dataloader))

# 从网络输出tensor中解析出SQUAD任务结果，代码参考自
# https://github.com/huggingface/transformers/blob/v3.1.0/examples/question-answering/run_squad.py
def get_results(features, example_indices, outputs):
    results = []
    for i, feature_index in enumerate(example_indices):
        eval_feature = features[feature_index.item()]
        unique_id = int(eval_feature.unique_id)
        output = [output[i].tolist() for output in outputs]
        start_logits, end_logits = output
        result = SquadResult(unique_id, start_logits, end_logits)
        results.append(result)
    return results

# 调用Model.deserialize_from_file接口完成将离线模型文件反序列化为Model类型实例。
mm_model = mm.Model()
assert mm_model.deserialize_from_file(OFFLINE_MODEL_PATH).ok()

with mm.System() as mm_sys:  # 初始化系统
    dev_count = mm_sys.device_count()
    print("Device count: ", dev_count)
    assert DEV_ID < dev_count
    # 打开MLU设备
    dev = mm.Device()
    dev.id = DEV_ID
    assert dev.active().ok()
    '''   
    print(mm_model.get_input_dimensions())
    print(mm_model.get_output_dimensions())
    print(mm_model.get_input_data_types())
    print(mm_model.get_output_data_types())
    '''
    engine = mm_model.create_i_engine()
    assert engine is not None
    # 创建Context
    context = engine.create_i_context()
    assert context is not None
    # 创建MLU任务队列
    queue = dev.create_queue()
    assert queue is not None
    # 创建输入tensor
    inputs = context.create_inputs()
    
    # 记录开始时间
    start_time = time.time()

    all_results = []
    for batch in tqdm(eval_dataloader, desc = "Evaluating"):
        batch = tuple(t for t in batch)
        # 准备输入数据
        for i in range(3):
            inputs[i].from_numpy(batch[i].numpy())
        # 向MLU下发任务
        outputs = []
        status = context.enqueue(inputs, outputs, queue)
        assert status.ok(), str(status)
        # 等待任务执行完成
        status = queue.sync()
        assert status.ok(), str(status)
        # 处理输出数据
        outputs_np = []
        for tensor in outputs:
            outputs_np.append(tensor.asnumpy())
        all_results.extend(get_results(features, batch[3], outputs_np))

    # 记录结束时间
    end_time = time.time()
    
    print("Final Time: %f seconds",end_time-start_time)


