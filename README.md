# Triton Python Backend  + Python MagicMind
## 获取代码
- git clone  https://github.com/xiaoqi25478/triton_python_backend_with_magicmind.git
- 论坛帖子:https://forum.cambricon.com/index.php?m=content&c=index&a=show&catid=176&id=2241

## 功能说明
- NVIDIA Triton推理服务是NVIDIA推出的开源推理框架，主要为用户提供在云和边缘推理上部署的解决方案,支持多种不同的后端，如Tensorflow,Pytorch,TensorTRT，Python等等。
- MagicMind 是面向寒武纪 MLU 的推理加速引擎,MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX,Caffe 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
- 通过Triton Python Backend+MagicMind Python版本的方式实现Bert_Case、YOLOv5n、resnet50_vd三个模型的推理服务化


## 环境介绍
- Device:MLU370 X8
- MagicMind版本 0.14.0
- Python:3.7

## 模型部署
**文件介绍**\
仓库主要包含bert resnet50_vd和yolov5n三个模型的triton+magicmind推理模型服务代码
三个模型代码内部结构、名称遵循一致规范。 \
因此以bert文件夹作为示例说明
- magicmind_codes 
    - codes 相关原始模型代码文件
    - data 模型数据集相关文件
    - models 存放原始模型文件(pt_model)已经转换生成的MagicMind模型文件(mm_model)
    - weights 模型权重相关文件
    - run_docker_magicmind_bert.sh magicmind docker运行脚本
    - magicmind_build.py MagicMind模型生成脚本
    - magicmind_build.sh 一键运行magicmind_build.py，生成不同精度的MagicMind模型
    - magicmind_infer.py MagicMind模型推理脚本
    - magicmind_build.sh 一键运行magicmind_infer.py
    - prepare_mm_env.sh MagicMind Docker镜像环境准备文件，需要进入Docker后运行
- triton_codes
    - mm_models triton模型文件 包含服务端model.py config.pbtxt和客户端client.py
    - start_triton_client.sh 一键运行client端脚本
    - start_triton_server.sh 一键运行server端脚本
    - start_triton_client_pref.sh 一键运行client性能测试脚本
    - client_requirements.txt client pip requirements文件，由prepare_client_env.sh调用
    - prepare_server(client)_env.sh 服务端(客户端)环境安装脚本，需进入相应docker后运行
    - run_docker_triton_server(client).sh 启动服务端(客户端)docker镜像

### 使用方法
#### Maigcimind
利用Magicmind实现模型转换与生成，已预先生成多种不同精度的模型存放在mm_model文件夹内，此步骤可以跳过。
- 启动MagicMind Docker环境
```
bash run_docker_magicmind_bert.sh
```
- 准备环境
```
bash prepare_mm_env.sh
```
- 生成不同精度的MagicMind Model
```
bash magicmind_build.sh
```
- MagicMind推理Demo
```
bash magicmind_infer.sh
```

#### Triton
利用Triton实现推理服务化
- Server端

**在mm_model/xx/1/model.py文件内可以指定特定精度的MagicMind模型**
```

# 运行Server Docker
bash run_docker_triton_server.sh

# Server端环境安装
bash prepare_server_env.sh

# Server端启动
bash start_triton_server.sh
```
- Client端

**在mm_model/xx/client.py文件内可以指定特定batch_size**
```
# Client Docker
bash run_docker_triton_client.sh

# Client端环境安装
bash prepare_client_env.sh

# Client端启动
bash start_triton_client.sh

# Client端可开启perf_client进行性能测试 bs从1到32
bash start_triton_client_pref.sh
```

## 精度与性能

**mm_build设置bs维度为可变**

**SEQ_LEN 128**

*MagicMind数据device:x8和mm版本0.14*

### Bert Case 
**精度** 
| Percision | Device | Best_exact(mm only) | Best_f1(mm only) |
| ------ | ------  | ------ | ------ |
| FP32 |  MLU370 X8  | 74.91(79.896) | 82.613(87.485) |
| FP16 |  MLU370 X8  | 74.92(79.139) | 82.625(86.977) | 

**性能**
| Percision | Device | Batch Size | Client p95 latency| Client p99 latency | Client Throughput | MagicMind FPS | MagicMind 平均响应时间/ms |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| FP16 |  MLU370 X8 | 1  | 4074 usec | 4243 usec | 247.198 infer/sec |	753.26 | 2.6391 |
| FP16 |  MLU370 X8 | 4  | 7181 usec | 7254 usec | 565.729 infer/sec | 1392.2| 5.73 |
| FP16 |  MLU370 X8 | 8  | 11033 usec | 11058 usec | 732.382 infer/sec |1605.3 | 9.9502 |
| FP16 |  MLU370 X8 | 16  | 20507 usec | 20627 usec | 795.478 infer/sec |2465.8 | 12.961 |
| FP16 |  MLU370 X8 | 32  | 36087 usec | 36379 usec | 890.589 infer/sec |2606.4| 24.537 |
| FP16 |  MLU370 X8 | 64  | 69288 usec | 69544 usec | 927.914 infer/sec |2674.2 | 47.848 |
| FP32 |  MLU370 X8 | 1  | 10600 usec | 11025 usec | 95.378 infer/sec |  172.43 | 11.58 |
| FP32 |  MLU370 X8 | 4  | 18585 usec | 18776 usec | 223.09 infer/sec | 406.22 | 19.674 |
| FP32 |  MLU370 X8 | 8  | 31002 usec | 31137 usec | 258.644 infer/sec | 440.48 | 36.305 |
| FP32 |  MLU370 X8 | 16  | 55321 usec | 55413 usec | 289.752 infer/sec | 474.82 | 67.375 |
| FP32 |  MLU370 X8 | 32  | 103358 usec | 103675 usec | 309.308 infer/sec |494.56 | 129.39 |
| FP32 |  MLU370 X8 | 64  | 198804 usec | 198918 usec | 319.974 infer/sec |	505.92 | 252.98 |

### YOLOv5n
**mm_build设置bs维度为可变**

*MagicMind数据device:x8和mm版本0.14*

**640*640**

**精度(基于coco2017 1000张测试图片)** 

| Percision | Device |  IoU=0.50:0.95(mm only) | IoU=0.50(mm only) |
| ------ | ------ |  ------ | ------ |
| FP32 |  MLU370 X8 | 0.226(0.226) | 0.375(0.375) |
| FP16 |  MLU370 X8 | 0.226(0.226) | 0.375(0.375) |
| INT8 |  MLU370 X8 | 0.226(0.226) | 0.375(0.375) |

**性能**

| Percision | Device | Batch Size | Client p95 latency| Client p99 latency | Client Throughput | MagicMind FPS | MagicMind 平均响应时间/ms |
| ------ | ------ | ------ | ------ | ------ | ------ |------ | ------ |
| INT8 |  MLU370 X8 | 1  | 8708 usec | 8787 usec | 115.768 infer/sec | 1955.06 | 1.0073 |
| INT8 |  MLU370 X8 | 4  | 11594 usec | 11786 usec | 349.968 infer/sec | 5813.4 | 1.3607 |
| INT8 |  MLU370 X8 | 8  | 23213 usec | 23472 usec | 350.631 infer/sec | 5884.8 | 2.7025 |
| INT8 |  MLU370 X8 | 16  | 51815 usec | 58614 usec | 311.969 infer/sec |5802.6| 5.4959 |
| INT8 |  MLU370 X8 | 32  | 111402 usec | 112418 usec | 289.748 infer/sec |5858.8 | 10.897 |
| INT8 |  MLU370 X8 | 64  | 223826 usec | 232501 usec | 291.527 infer/sec |6108.2| 20.897 |
| FP16 |  MLU370 X8 | 1  | 9007 usec | 9156 usec | 113.045 infer/sec |1724.24	| 1.1443 |
| FP16 |  MLU370 X8 | 4  | 12724 usec | 13150 usec | 333.3 infer/sec |4756.8 | 1.6644 |
| FP16 |  MLU370 X8 | 8  | 23474 usec | 23474 usec | 346.189 infer/sec |4698.2 | 3.3883 |
| FP16 |  MLU370 X8 | 16  | 52690 usec | 19981 usec | 306.637 infer/sec |4800.8 | 6.6462 |
| FP16 |  MLU370 X8 | 32  | 114447 usec | 116024 usec | 280.859 infer/sec |4792 | 13.326 |
| FP16 |  MLU370 X8 | 64  | 225787 usec | 234397 usec | 284.416 infer/sec |4864 | 26.214 |
| FP32 |  MLU370 X8 | 1  | 9653 usec | 9806 usec | 109.932 infer/sec |978.56 | 2.0281 |
| FP32 |  MLU370 X8 | 4  | 14796 usec | 15104 usec | 289.74 infer/sec |2066.2 | 3.8554 |
| FP32 |  MLU370 X8 | 8  | 25832 usec | 27707 usec | 336.849 infer/sec |2050.4| 7.7853 |
| FP32 |  MLU370 X8 | 16  | 54133 usec | 57325 usec | 301.304 infer/sec |925.9 | 34.518 |
| FP32 |  MLU370 X8 | 32  | 128916 usec | 129968 usec | 271.975 infer/sec |925.9| 29.944|
| FP32 |  MLU370 X8 | 64  | 225119 usec | 231401 usec | 287.974 infer/sec |2154.6 | 59.293 |

### resnet50_vd
**mm_build设置bs维度为可变**
**Renet50模型来源自paddlepaddle**

*MagicMind数据device:x8和mm版本0.14*

**224*224**

**精度(基于ImageNet2012 1000张测试图片)** 
| Percision | Device | Top1(mm only) | Top5(mm only) |
| ------ | ------ |  ------ | ------ |
| FP32 |  MLU370 X8 |  0.7897(0.7907) | 0.9589(0.9589) |
| FP16 |  MLU370 X8 |  0.7897(0.7907) | 0.9589(.9589)  |
| INT8 |  MLU370 X8 |  0.7927(0.7907) | 0.9549(0.9549) |

**性能**
| Percision | Device | Batch Size | Client p95 latency| Client p99 latency | Client Throughput |  MagicMind FPS | MagicMind 平均响应时间/ms |
| ------ | ------ | ------ | ------ | ------ | ------ |------ | ------ |
| INT8 |  MLU370 X8 | 1  | 2199 usec | 3770 usec | 498.721 infer/sec | 2760.2 | 0.70897 |
| INT8 |  MLU370 X8 | 4  | 4501 usec | 9498 usec | 1086.51 infer/sec |9039.8 | 0.86944 |
| INT8 |  MLU370 X8 | 8  | 7632 usec | 15105 usec | 1336.74 infer/sec |10746 | 1.4733 |
| INT8 |  MLU370 X8 | 16  | 14143 usec | 58614 usec | 1429.19 infer/sec |11980.2 | 2.6548 |
| INT8 |  MLU370 X8 | 32  | 39271 usec | 50982 usec | 1094.99 infer/sec |11768.4 | 5.4198 |
| INT8 |  MLU370 X8 | 64  | 90591 usec | 105244 usec | 938.385 infer/sec |12279 | 10.392 |
| FP16 |  MLU370 X8 | 1  | 3347 usec | 7711 usec | 346.49 infer/sec |1494.9 | 1.3224 |
| FP16 |  MLU370 X8 | 4  | 5653 usec | 12201 usec | 865.789 infer/sec |4460 | 1.7781 |
| FP16 |  MLU370 X8 | 8  | 11056 usec | 21661 usec | 1012.31 infer/sec |5467 | 2.9104 |
| FP16 |  MLU370 X8 | 16  | 21374 usec | 35280 usec | 1137.64 infer/sec |6152.4 | 5.1837 |
| FP16 |  MLU370 X8 | 32  | 49424 usec | 62083 usec | 927.888 infer/sec |6622.2 | 9.6423 |
| FP16 |  MLU370 X8 | 64  | 99798 usec | 115607 usec | 839.011 infer/sec |6648.2 | 19.218 |
| FP32 |  MLU370 X8 | 1  | 8762 usec | 12943 usec | 147.927 infer/sec |577.38 | 3.448 |
| FP32 |  MLU370 X8 | 4  | 12023 usec | 18161 usec | 428.614 infer/sec |1487.14 | 5.3612 |
| FP32 |  MLU370 X8 | 8  | 19086 usec | 28850 usec | 527.055 infer/sec |1781.04 | 8.9637 |
| FP32 |  MLU370 X8 | 16  | 37026 usec | 49594 usec | 571.367 infer/sec |1790.26 | 17.853 |
| FP32 |  MLU370 X8 | 32  | 75573 usec | 84856 usec | 536.824 infer/sec |1917.18 | 33.35 |
| FP32 |  MLU370 X8 | 64  | 148240 usec | 163989 usec | 515.379 infer/sec |2027.8 | 63.075 |


## 遇到问题记录
Q: Pref Client: No valid requests recorded within time interval. Please use a larger time window

A: 可能需要设置确定的输入形状 不再是可变形状 (后经验证，取决于自己的模型，可以设置为可变) \
参考链接:https://github.com/triton-inference-server/server/issues/217  \
对于Bert类模型,还需要设置输入为0 (后经验证，必须要为0)
否则 报错cnrt信息如下:
```
2022-10-21 09:21:39.667489: [cnrtError] [12905] [Card : 0] Error occurred during calling 'cnQueueSync' in CNDrv interface.
2022-10-21 09:21:39.667513: [cnrtError] [12905] [Card : 0] Return value is 100124, CN_INVOKE_ERROR_ADDRESS_SPACE, means that "operation not supported on global/shared address space"
2022-10-21 09:21:39.667521: [cnrtError] [12905] [Card : 0] cnrtQueueSync: MLU queue sync failed.
```
参考链接:https://github.com/triton-inference-server/server/issues/819
