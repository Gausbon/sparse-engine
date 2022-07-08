import os
import numpy as np

from utils import conv_data_to_sparse, approximate_float

class File_writer():
    def __init__(self, func_path="../src/Cor/Src/quantization_inference.c", 
        data_path="../src/Cor/Inc/data.h"):
        self.func_path = func_path
        self.data_path = data_path
        self.func_file = os.open(self.func_path, "w")
        self.data_file = os.open(self.data_path, "w")


    def __del__(self):
        self.func_file.close()
        self.data_file.close()


    def write_file(self, r_file, w_file):
        lines = r_file.readlines()
        for item in lines:
            w_file.print(item)


    def write_tensor(self, tensor, name, is_static, data_type):
        if (is_static):
            self.data_file.print("static ")
        self.data_file.print(data_type + ' ' + name + '[' + str(tensor.size+1) + '] = {')
        tensor_flatten = tensor.flatten()
        for i in range(0, tensor_flatten.size):
            self.data_file.write(tensor_flatten[i])
            if (i != tensor_flatten.size - 1):
                self.data_file.write(',')
        self.data_file.write('};\n\n')


    def write_func_call(self, name, param_list):
        self.func_file.write('   ' + name + '(')
        for i in range(0, len(param_list)):
            self.func_file.write(param_list[i])
            if (i != len(param_list)-1):
                self.func_file.write(',')
        self.func_file.write(');\n\n')


    def write_param_parser(self, name, param_dict, func_file):
        for key, value in param_dict.items():
            func_file.write('   ' + name + '.' + key + '=' + str(value) + ';\n')
        func_file.write('\n')



class Model_deployment():
    def __init__(self, tensor_path="../../TinySPOS/tensor"):
        self.tensor_path = tensor_path
        self.downsample_list_dict = {}
        self.mv2block_list_dict = {}
        self.transformer_list_dict = {}
        self.last_conv_dict = {}
        self.pooling_dict = {}
        self.classifier_dict = {}
        self.counter = 0
        self.file_writer = File_writer()


    def load_tensor(self):
        tensor_dir = os.listdir(self.tensor_path)
        # first round: downsample
        
        for file in tensor_dir:
            name_list = file.split('.')
            if (name_list[0] == "downsample"):
                if (name_list[1] in self.downsample_list_dict.keys()):
                    block_dict = self.downsample_list_dict[name_list[1]]
                else:
                    block_dict = {}
                    self.downsample_list_dict[name_list[1]] = block_dict
            elif (name_list[0] == "mv2block"):
                if (name_list[1] in self.mv2block_list_dict.keys()):
                    block_dict = self.mv2block_list_dict[name_list[1]]
                else:
                    block_dict = {}
                    self.mv2block_list_dict[name_list[1]] = block_dict
            elif (name_list[0] == "transformer"):
                if (name_list[1] in self.transformer_list_dict.keys()):
                    block_dict = self.transformer_list_dict[name_list[1]]
                else:
                    block_dict = {}
                    self.transformer_list_dict[name_list[1]] = block_dict
            elif (name_list[0] == "qlast_conv"):
                block_dict = self.last_conv_dict
            elif (name_list[0] == "qglobal_pooling" or name_list[0] == "global_pooling" ):
                block_dict = self.pooling_dict
            elif (name_list[0] == "qclassifier"):
                block_dict = self.classifier_dict
            else:
                print("Cannot resolve " + file)
                continue

            # keyname: {block_type}.{block_name}
            key_name = name_list[2]
            for i in range(3, len(name_list) - 1):
                key_name = key_name + '.' + name_list[i]
            value = np.load(self.tensor_path + "/" + file)
            if (value.size == 1):
                value = int(value)
            block_dict[key_name] = value
        
        # sort by the index: {blockname}_{index}
        self.downsample_list_dict = dict(sorted(self.downsample_list_dict.items(), key=lambda x: x[0].split('_')[-1]))
        self.mv2block_list_dict = dict(sorted(self.mv2block_list_dict.items(), key=lambda x: x[0].split('_')[-1]))
        self.transformer_list_dict = dict(sorted(self.transformer_list_dict.items(), key=lambda x: x[0].split('_')[-1]))


    def deploy_add(self, name, block_dict, input_section_1, input_section_2, output_section, count):
        # init
        param_list = ['&' + input_section_1, '&' + input_section_2]

        # quant param
        qi1_scale = block_dict[name + '.qmatmul_qk.qi1.scale']
        qi1_mult, qi1_shift = approximate_float(qi1_scale)
        qi2_scale = block_dict[name + '.qmatmul_qk.qi2.scale']
        qi2_mult, qi2_shift = approximate_float(qi2_scale)
        qo_scale = block_dict[name + '.qmatmul_qk.qo.scale']
        qo_mult, qo_shift = approximate_float(qo_scale)
        qi1_offset = -block_dict[name + '.qmatmul_qk.qi1.zero_point']
        qi2_offset = -block_dict[name + '.qmatmul_qk.qi2.zero_point']
        qo_offset = block_dict[name + '.qmatmul_qk.qo.zero_point']
        
        # func call
        param_list.extend([qi1_offset, qi1_mult, qi1_shift, qi2_offset, qi2_mult, qi2_shift,
            0, ('&' + output_section), qo_offset, qo_mult, qo_shift, count])
        self.file_writer.write_func_call('arm_elementwise_add_s8', param_list)


    def deploy_self_atten(self, name, block_dict, input_section, output_section,
                    self_atten_dict, dims_dict):
        # init
        param_list = ['&ctx', '&self_atten_params', '&t_quant_params', '&t_quant_params_1',
        '&input_dims', ('&' + input_section), ('&' + output_section)]
        self.file_writer.write_param_parser('input_dims', dims_dict)

        # parse qkv param
        qkv_qi1_scale = block_dict[name + '.qmatmul_qk.qi1.scale']
        qkv_qi2_scale = block_dict[name + '.qmatmul_qk.qi2.scale']
        qkv_qo_scale = block_dict[name + '.qmatmul_qk.qo.scale']        
        qkv_qi1_offset = block_dict[name + '.qmatmul_qk.qi1.zero_point']
        qkv_qi2_offset = block_dict[name + '.qmatmul_qk.qi2.zero_point']
        qkv_qo_offset = block_dict[name + '.qmatmul_qk.qo.zero_point']
        M = qkv_qi1_scale * qkv_qi2_scale / qkv_qo_scale
        qkv_mult, qkv_shift = approximate_float(M)

        self_atten_dict['qk_multiplier'] = qkv_mult
        self_atten_dict['qk_shift'] = qkv_shift
        self_atten_dict['qk_qi1_offset'] = -qkv_qi1_offset
        self_atten_dict['qk_qi2_offset'] = -qkv_qi2_offset
        self_atten_dict['qk_qo_offset'] = qkv_qo_offset

        attnv_qi1_scale = block_dict[name + '.qmatmul_attnv.qi1.scale']
        attnv_qi2_scale = block_dict[name + '.qmatmul_attnv.qi2.scale']
        attnv_qo_scale = block_dict[name + '.qmatmul_attnv.qo.scale']
        attnv_qi1_offset = block_dict[name + '.qmatmul_attnv.qi1.zero_point']
        attnv_qi2_offset = block_dict[name + '.qmatmul_attnv.qi2.zero_point']
        attnv_qo_offset = block_dict[name + '.qmatmul_attnv.qo.zero_point']
        M = attnv_qi1_scale * attnv_qi2_scale / attnv_qo_scale
        attnv_mult, attnv_shift = approximate_float(M)

        self_atten_dict['attnv_multiplier'] = attnv_mult
        self_atten_dict['attnv_shift'] = attnv_shift
        self_atten_dict['attnv_qi1'] = -attnv_qi1_offset
        self_atten_dict['attnv_qi2'] = -attnv_qi2_offset
        self_atten_dict['attnv_qo'] = attnv_qo_offset
        
        # softmax
        softmax_qi_scale = block_dict[name + '.qsoftmax1.qi.scale']
        softmax_qi_offset = block_dict[name + '.qsoftmax1.qi.zero_point']
        softmax_qi_mult, softmax_qi_shift = approximate_float(softmax_qi_scale)
        self_atten_dict['softmax_qi_mult'] = softmax_qi_mult
        self_atten_dict['softmax_qi_shift'] = softmax_qi_shift
        self_atten_dict['softmax_qi_offset'] = -softmax_qi_offset
        softmax_qo_scale = block_dict[name + '.qsoftmax1.qo.scale']
        softmax_qo_offset = block_dict[name + '.qsoftmax1.qo.zero_point']
        softmax_qo_mult, softmax_qo_shift = approximate_float(softmax_qo_scale)
        self_atten_dict['softmax_qo_mult'] = softmax_qo_mult
        self_atten_dict['softmax_qo_shift'] = softmax_qo_shift
        self_atten_dict['softmax_qo_offset'] = softmax_qo_offset

        # func call
        self.file_writer.write_param_parser('self_atten_params', self_atten_dict)
        self.file_writer.write_func_call('arm_self_attention_s8', param_list)


    def deploy_conv(self, name, block_dict, is_sparse, is_depthwise,
                    input_section, output_section, conv_dict, input_dict, filter_dict):
        # init
        param_list = ['&ctx']
        if (is_depthwise):
            param_list.append('&dw_conv_params')
        else:
            param_list.append('&conv_params')
        param_list.extend(['&c_quant_params', '&input_dims', ('&' + input_section)])
        self.write_param_parser('input_dims', input_dict)
        self.write_param_parser('filter_dims', filter_dict)

        # weight operation
        weight = block_dict[name + '.conv_module.weight'].transpose((0, 2, 3, 1))
        if (is_sparse):
            weight = conv_data_to_sparse(input)
        weight_name = 'weight_' + self.counter
        param_list.extend(['&filter_dims', ('&' + weight_name)])
        self.file_writer.write_tensor(weight, weight_name, True, 'q7_t')
        
        # bias operation
        bias = block_dict[name + '.conv_module.bias']
        bias_name = 'bias_' + self.counter
        param_list.extend(['&bias_dims', ('&' + bias_name)])
        self.file_writer.write_tensor(bias, bias_name, True, 'q7_t')

        # output operation and parse conv params
        param_list.extend(['&output_dims',('&' + output_section)])
        conv_dict['activation.max'] = block_dict[name + '.qw.max']
        conv_dict['activation.min'] = block_dict[name + '.qw.min']
        # notice the input offset is negative zero point
        conv_dict['input_offset'] = -block_dict[name + '.qi.zero_point']
        conv_dict['output_offset'] = block_dict[name + '.qo.zero_point']
        if (is_depthwise):
            self.file_writer.write_param_parser('dw_conv_params', conv_dict)
        else:
            self.file_writer.write_param_parser('conv_params', conv_dict)

        # parse quant params
        qi_scale = block_dict[name + '.qi.scale']
        qo_scale = block_dict[name + '.qo.scale']
        qw_scale = block_dict[name + '.qw.scale']
        M = qi_scale * qw_scale / qo_scale
        mult, shift = approximate_float(M)
        mult_name = 'mult_' + self.counter
        shift_name = 'shift_' + self.counter
        self.file_writer.write_tensor(mult, mult_name, True, 'q31_t')
        self.file_writer.write_tensor(shift, shift_name, True, 'q31_t')
        
        quant_dict = {}
        quant_dict['multiplier'] = '&' + mult_name
        quant_dict['shift'] = '&' + shift_name
        self.write_param_parser('c_quant_params', quant_dict)

        if (is_sparse):
            param_list.append(weight.size)

        # function call generate
        func_name = 'arm_'
        if (is_depthwise):
            func_name = func_name + 'depthwise_conv_s8'
        else:
            func_name = func_name + 'convolve_s8'
        
        if (is_sparse):
            func_name = func_name + '_sparse'
        self.file_writer.write_func_call(func_name, param_list)

        self.counter += 1


    def deploy_linear(self, name, block_dict, is_sparse,
                    input_section, output_section, fc_dict):
        # init
        param_list = ['&ctx', '&fc_params', '&t_quant_params', '&input_dims', ('&' + input_section)]
        
        # weight operation
        weight = block_dict[name + '.fc_module.weight']
        if (is_sparse):
            weight = conv_data_to_sparse(input)
        weight_name = 'weight_' + self.counter
        param_list.extend(['&filter_dims', ('&' + weight_name)])
        self.file_writer.write_tensor(weight, weight_name, True, 'q7_t')
        
        # bias operation
        bias = block_dict[name + '.fc_module.bias']
        bias_name = 'bias_' + self.counter
        param_list.extend(['&bias_dims', ('&' + bias_name)])
        self.file_writer.write_tensor(bias, bias_name, True, 'q7_t')

        # output operation and parsefc params
        param_list.extend(['&output_dims',('&' + output_section)])
        fc_dict['activation.max'] = block_dict[name + '.qw.max']
        fc_dict['activation.min'] = block_dict[name + '.qw.min']
        # notice the input offset is negative zero point
        fc_dict['input_offset'] = -block_dict[name + '.qi.zero_point']
        fc_dict['output_offset'] = block_dict[name + '.qo.zero_point']
        self.file_writer.write_param_parser('fc_params', fc_dict)

        # parse quant params
        qi_scale = block_dict[name + '.qi.scale']
        qo_scale = block_dict[name + '.qo.scale']
        qw_scale = block_dict[name + '.qw.scale']
        M = qi_scale * qw_scale / qo_scale
        mult, shift = approximate_float(M)

        quant_dict = {}
        quant_dict['multiplier'] = int(mult)
        quant_dict['shift'] = int(shift)
        self.file_writer.write_param_parser('t_quant_params', quant_dict)

        if (is_sparse):
            param_list.append(weight.size)

        # function call generate
        func_name = 'arm_fully_connected_s8'
        if (is_sparse):
            func_name = func_name + '_sparse'
        self.file_writer.write_func_call(func_name, param_list)

        self.counter += 1


    def deploy_mv2block(self, name, block_dict, input_dict, filter_dict, conv_dict):
        type_name = name.split('_')[-2]
        if (type_name.endswith('0')):
            expansion = 2
            stride = 1
        elif (type_name.endswith('1')):
            expansion = 3
            stride = 1
        elif (type_name.endswith('2')):
            expansion = 4
            stride = 1
        elif (type_name.endswith('block')):
            expansion = 2
            stride = 2
        else:
            print('Cannot parse block as mv2block: ' + name)
            return

        input_dict['N' = ]
        


    def print_tensor(self):
        for key, value in self.downsample_list_dict.items():
            print("block: " + key)
            for key0, value0 in value.items():
                print(("key: " + key0).ljust(40, ' '), end="")
                print("shape: ", end="")
                print(value0.shape)
        for key, value in self.mv2block_list_dict.items():
            print("block: " + key)
            for key0, value0 in value.items():
                print(("key: " + key0).ljust(40, ' '), end="")
                print("shape: ", end="")
                print(value0.shape)
        for key, value in self.transformer_list_dict.items():
            print("block: " + key)
            for key0, value0 in value.items():
                print(("key: " + key0).ljust(40, ' '), end="")
                print("shape: ", end="")
                print(value0.shape)
        print("block: last_conv")
        for key, value in self.last_conv_dict.items():
            print(("key: " + key).ljust(40, ' '), end="")
            print("shape: ", end="")
            print(value.shape)
        print("block: pooling")
        for key, value in self.pooling_dict.items():
            print(("key: " + key).ljust(40, ' '), end="")
            print("shape: ", end="")
            print(value.shape)
        print("block: classifier")
        for key, value in self.classifier_dict.items():
            print(("key: " + key).ljust(40, ' '), end="")
            print("shape: ", end="")
            print(value.shape)


if __name__ == '__main__':
    print("start")
    model_deployment = Model_deployment()
    model_deployment.load_tensor()
    model_deployment.print_tensor()