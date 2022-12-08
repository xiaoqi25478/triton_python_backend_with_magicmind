from tritonclient.utils import *
import tritonclient.http as httpclient
import sys
import time
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer,squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV1Processor
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate
import numpy as np
import os 

cur_dir = os.path.dirname(os.path.abspath(__file__))

model_name = "bert_case"

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

# 模型单次推理批大小
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 128

# 加载数据
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", do_lower_case = False)
squad_processor = SquadV1Processor()
examples = squad_processor.get_dev_examples("", filename=os.path.join(cur_dir,
                                            "../../../magicmind_codes/data/dev-v1.1.json"))
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
# 若mm model为固定bs 请将drop_last设置为true
eval_dataloader = DataLoader(dataset, sampler = eval_sampler, batch_size = BATCH_SIZE, drop_last = False)
print("Num examples = ", len(dataset))
print("Batch size = ", BATCH_SIZE)
print("Iterations = ", len(eval_dataloader))

with httpclient.InferenceServerClient("localhost:8000") as client:
    start_time = time.time()
    all_results = []
    for batch in tqdm(eval_dataloader, desc = "Evaluating"):
        batch = tuple(t for t in batch)
        input0_data = batch[0].numpy().astype(np.int32)
        input1_data = batch[1].numpy().astype(np.int32)
        input2_data = batch[2].numpy().astype(np.int32)
        
        inputs = [
            httpclient.InferInput("INPUT0", input0_data.shape,
                                np_to_triton_dtype(input0_data.dtype)),
            httpclient.InferInput("INPUT1", input1_data.shape,
                                np_to_triton_dtype(input1_data.dtype)),
            httpclient.InferInput("INPUT2", input2_data.shape,
                            np_to_triton_dtype(input2_data.dtype)),
        ]

        inputs[0].set_data_from_numpy(input0_data)
        inputs[1].set_data_from_numpy(input1_data)
        inputs[2].set_data_from_numpy(input2_data)
        
        outputs = [
            httpclient.InferRequestedOutput("OUTPUT0"),
            httpclient.InferRequestedOutput("OUTPUT1"),
        ]

        response = client.infer(model_name,
                                inputs,
                                request_id=str(1),
                                outputs=outputs)

        result = response.get_response()
        output0_data = response.as_numpy("OUTPUT0")
        output1_data = response.as_numpy("OUTPUT1")
        outputs_np = [output0_data,output1_data]
        all_results.extend(get_results(features, batch[3], outputs_np))
    # 记录结束时间
    end_time = time.time()
    print("Final Time: %f seconds"%(end_time-start_time))

    # 计算精度
    print("Computing Acc....")
    import os 
    if not os.path.exists("bert_output"):
        os.makedirs("bert_output")
    output_prediction_file = "output/predictions.json"
    output_nbest_file = "output/nbest_predictions.json"

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        20,  # n best size
        30,  # max answer length
        False,  # do lower case
        output_prediction_file,
        output_nbest_file,
        None,
        False,
        False,
        0.0,
        tokenizer
    )

    squad_acc = squad_evaluate(examples, predictions)
    print("SQUAD results: {}".format(squad_acc))