from sys import int_info

import numpy as np
import triton_python_backend_utils as pb_utils
import magicmind.python.runtime as mm
import json
import os 

cur_dir = os.path.dirname(os.path.abspath(__file__))

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """`auto_complete_config` is called only once when the server is started
        with `--strict-model-config=false`. Implementing this function is optional.
        A no implementation of `auto_complete_config` will do nothing. This function
        can be used to set `max_batch_size`, `input` and `output` properties of the
        model using `set_max_batch_size`, `add_input`, and `add_output`.
        These properties will allow Triton to load the model with minimal model
        configuration in absence of a configuration file. This function returns the
        `pb_utils.ModelConfig` object with these properties. You can use `as_dict`
        function to gain read-only access to the `pb_utils.ModelConfig` object.
        The `pb_utils.ModelConfig` object being returned from here will be used as
        the final configuration for the model.

        Note: The Python interpreter used to invoke this function will be destroyed
        upon returning from this function and as a result none of the objects created
        here will be available in the `initialize`, `execute`, or `finalize` functions.

        Parameters
        ----------
        auto_complete_model_config : pb_utils.ModelConfig
          An object containing the existing model configuration. You can build upon
          the configuration given by this object when setting the properties for 
          this model.

        Returns
        -------
        pb_utils.ModelConfig
          An object containing the auto-completed model configuration
        """
        inputs = [{
            'name': 'INPUT0',
            'data_type': 'TYPE_FP32',
            'dims': [4]
        }, {
            'name': 'INPUT1',
            'data_type': 'TYPE_FP32',
            'dims': [4]
        }]
        outputs = [{
            'name': 'OUTPUT0',
            'data_type': 'TYPE_FP32',
            'dims': [4]
        }, {
            'name': 'OUTPUT1',
            'data_type': 'TYPE_FP32',
            'dims': [4]
        }]

        # Demonstrate the usage of `as_dict`, `add_input`, `add_output`,
        # `set_max_batch_size`, and `set_dynamic_batching` functions.
        # Store the model configuration as a dictionary.
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config['input']:
            input_names.append(input['name'])
        for output in config['output']:
            output_names.append(output['name'])

        for input in inputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_input` will check for conflicts and
            # raise errors if an input with the same name already exists in
            # the configuration but has different data_type or dims property.
            if input['name'] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_output` will check for conflicts and
            # raise errors if an output with the same name already exists in
            # the configuration but has different data_type or dims property.
            if output['name'] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_max_batch_size(0)
        
        # To enable a dynamic batcher with default settings, you can use 
        # auto_complete_model_config set_dynamic_batching() function. It is 
        # commented in this example because the max_batch_size is zero.
        #
        # auto_complete_model_config.set_dynamic_batching()

        return auto_complete_model_config

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        '''
        在该函数内部主要进行MLU设备的初始化
        '''
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")

        # Get OUTPUT1 configuration
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT1")
        
        print('Initialized...')
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])
        
        self.DEV_ID = 0 #默认使用0号卡进行推理
        # 请选择自己要使用通的MagicMind Model
        self.MM_MODEL_PATH = os.path.join(
            cur_dir,
            "../../../../magicmind_codes/models/mm_model/bert_base_cased_squad_fp16.mm")
        print(self.MM_MODEL_PATH)
        if not os.path.exists(self.MM_MODEL_PATH):
            print("MM Model Not exist!")
            return 
        self.mm_model = mm.Model()
        assert self.mm_model.deserialize_from_file(self.MM_MODEL_PATH).ok()
        # 初始化MLU DEVICE
        with mm.System() as mm_sys:  # 初始化系统
            dev_count = mm_sys.device_count()
            print("Device count: ", dev_count)
            assert self.DEV_ID < dev_count
            # 打开MLU设备
            self.dev = mm.Device()
            self.dev.id = self.DEV_ID
            assert self.dev.active().ok()
            self.engine = self.mm_model.create_i_engine()
            assert self.engine is not None
            # 创建self.context
            self.context = self.engine.create_i_context()
            assert self.context is not None
            # 创建MLU任务队列
            self.queue = self.dev.create_queue()
            assert self.queue is not None
            # 创建输入tensor
            self.inputs = self.context.create_inputs()        
            
    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        '''
        在该函数内部主要进行推理请求的计算
        '''

        responses = []
        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        for request in requests:
            # Perform inference on the request and append it to responses list...
            # 获取输入
            for i in range(3):
                in_i = pb_utils.get_input_tensor_by_name(request, "INPUT"+str(i))
                self.inputs[i].from_numpy(in_i.as_numpy())
        
            # 执行推理
            outputs = []
            status = self.context.enqueue(self.inputs, outputs, self.queue)
            assert status.ok(), str(status)
            status = self.queue.sync()
            assert status.ok(), str(status)
            
            outputs_np = []
            for tensor in outputs:
                outputs_np.append(tensor.asnumpy())
                
            out_tensor_0 = pb_utils.Tensor("OUTPUT0",
                                           outputs_np[0].astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("OUTPUT1",
                                           outputs_np[1].astype(output1_dtype))

            inference_response  =   pb_utils.InferenceResponse(
                                    output_tensors=[out_tensor_0, out_tensor_1])
            responses.append(inference_response)

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
