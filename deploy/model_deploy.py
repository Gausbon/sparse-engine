import os
import numpy as np
import yaml
from utils import conv_data_to_sparse, approximate_float

class File_writer():
    def __init__(self, func_path, data_path):
        self.func_path = func_path
        self.data_path = data_path
        self.func_file = open(self.func_path,'w')
        self.data_file = open(self.data_path,'w')


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
            # tensor must be int
            self.data_file.write(int(tensor_flatten[i]))
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
    def __init__(self):
        with open('config.yml', 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        # basic info
        self.tensor_path = self.config['tensor_path']
        self.file_writer = File_writer(self.config['data_path'], self.config['func_path'])
        
        # dict list
        self.downsample_list_dict = {}
        self.mv2block_list_dict = {}
        self.transformer_list_dict = {}
        self.last_conv_dict = {}
        self.pooling_dict = {}
        self.classifier_dict = {}
        
        # static dict
        self.input_dict = {}        # input_dims
        self.filter_dict = {}       # filter_dims
        self.output_dict = {}       # filter_dims
        self.conv_dict = {}         # conv_params
        self.fc_dict = {}           # fc_params
        self.norm_dict = {}         # norm_params
        self.self_atten_dict = {}   # self_atten_params
        self.t_quant_dict = {}      # per_tensor
        self.c_quant_dict = {}      # per_channel

        # block counter for conv and linear
        self.counter = 0
        
        
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
                value = float(value)
            block_dict[key_name] = value
        
        # sort by the index: {blockname}_{index}
        self.downsample_list_dict = dict(sorted(self.downsample_list_dict.items(), 
                key=lambda x: x[0].split('_')[-1]))
        self.mv2block_list_dict = dict(sorted(self.mv2block_list_dict.items(), 
                key=lambda x: x[0].split('_')[-1]))
        self.transformer_list_dict = dict(sorted(self.transformer_list_dict.items(), 
                key=lambda x: x[0].split('_')[-1]))


    def deploy_add(self, name, block_dict, 
            input_section_1, input_section_2, output_section, count):
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


    def deploy_self_atten(self, name, block_dict, head_nums,
            input_section, output_section):
        # init
        param_list = ['&ctx', head_nums, '&input_dims', ('&' + input_section), ('&' + output_section)]
        self.file_writer.write_param_parser('input_dims', self.input_dict)

        # parse qkv param
        qkv_qi1_scale = block_dict[name + '.qmatmul_qk.qi1.scale']
        qkv_qi2_scale = block_dict[name + '.qmatmul_qk.qi2.scale']
        qkv_qo_scale = block_dict[name + '.qmatmul_qk.qo.scale']        
        qkv_qi1_offset = block_dict[name + '.qmatmul_qk.qi1.zero_point']
        qkv_qi2_offset = block_dict[name + '.qmatmul_qk.qi2.zero_point']
        qkv_qo_offset = block_dict[name + '.qmatmul_qk.qo.zero_point']
        M = qkv_qi1_scale * qkv_qi2_scale / qkv_qo_scale
        qkv_mult, qkv_shift = approximate_float(M)

        self.self_atten_dict['qk_multiplier'] = qkv_mult
        self.self_atten_dict['qk_shift'] = qkv_shift
        self.self_atten_dict['qk_qi1_offset'] = -qkv_qi1_offset
        self.self_atten_dict['qk_qi2_offset'] = -qkv_qi2_offset
        self.self_atten_dict['qk_qo_offset'] = qkv_qo_offset

        attnv_qi1_scale = block_dict[name + '.qmatmul_attnv.qi1.scale']
        attnv_qi2_scale = block_dict[name + '.qmatmul_attnv.qi2.scale']
        attnv_qo_scale = block_dict[name + '.qmatmul_attnv.qo.scale']
        attnv_qi1_offset = block_dict[name + '.qmatmul_attnv.qi1.zero_point']
        attnv_qi2_offset = block_dict[name + '.qmatmul_attnv.qi2.zero_point']
        attnv_qo_offset = block_dict[name + '.qmatmul_attnv.qo.zero_point']
        M = attnv_qi1_scale * attnv_qi2_scale / attnv_qo_scale
        attnv_mult, attnv_shift = approximate_float(M)

        self.self_atten_dict['attnv_multiplier'] = attnv_mult
        self.self_atten_dict['attnv_shift'] = attnv_shift
        self.self_atten_dict['attnv_qi1'] = -attnv_qi1_offset
        self.self_atten_dict['attnv_qi2'] = -attnv_qi2_offset
        self.self_atten_dict['attnv_qo'] = attnv_qo_offset
        
        # softmax
        softmax_qi_scale = block_dict[name + '.qsoftmax1.qi.scale']
        softmax_qi_offset = block_dict[name + '.qsoftmax1.qi.zero_point']
        softmax_qi_mult, softmax_qi_shift = approximate_float(softmax_qi_scale)
        self.self_atten_dict['softmax_qi_mult'] = softmax_qi_mult
        self.self_atten_dict['softmax_qi_shift'] = softmax_qi_shift
        self.self_atten_dict['softmax_qi_offset'] = -softmax_qi_offset
        softmax_qo_scale = block_dict[name + '.qsoftmax1.qo.scale']
        softmax_qo_offset = block_dict[name + '.qsoftmax1.qo.zero_point']
        softmax_qo_mult, softmax_qo_shift = approximate_float(softmax_qo_scale)
        self.self_atten_dict['softmax_qo_mult'] = softmax_qo_mult
        self.self_atten_dict['softmax_qo_shift'] = softmax_qo_shift
        self.self_atten_dict['softmax_qo_offset'] = softmax_qo_offset

        # func call
        self.file_writer.write_param_parser('self_atten_params', self.self_atten_dict)
        self.file_writer.write_func_call('arm_self_attention_s8', param_list)


    def deploy_conv(self, name, block_dict, is_sparse, is_depthwise,
            input_section, output_section):
        # init
        param_list = ['&ctx']
        if (is_depthwise):
            param_list.append('&dw_conv_params')
        else:
            param_list.append('&conv_params')
        param_list.extend(['&c_quant_params', '&input_dims', ('&' + input_section)])
        self.file_writer.write_param_parser('input_dims', self.input_dict)
        self.file_writer.write_param_parser('filter_dims', self.filter_dict)
        self.file_writer.write_param_parser('output_dims', self.output_dict)

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
        self.conv_dict['activation.max'] = block_dict[name + '.qw.max']
        self.conv_dict['activation.min'] = block_dict[name + '.qw.min']
        # notice the input offset is negative zero point
        self.conv_dict['input_offset'] = -block_dict[name + '.qi.zero_point']
        self.conv_dict['output_offset'] = block_dict[name + '.qo.zero_point']
        if (is_depthwise):
            self.file_writer.write_param_parser('dw_conv_params', self.conv_dict)
        else:
            self.file_writer.write_param_parser('conv_params', self.conv_dict)

        # parse quant params
        M = block_dict[name + 'M']
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

    def deploy_pooling(self, name, block_dict, is_avg,
            input_section, output_section):
        # init
        param_list = ['&ctx', '&pool_params', '&input_dims', ('&' + input_section), 
                '&filter_dims', '&output_dims']
        
        self.file_writer.write_param_parser('input_dims', self.input_dict)
        self.file_writer.write_param_parser('filter_dims', self.filter_dict)
        self.file_writer.write_param_parser('output_dims', self.output_dict)

        # get input offset
        param_list.append(-block_dict[name + '.qi.zero_point'])
        
        # get output offset
        param_list.append(block_dict[name + '.qo.zero_point'])

        # get input mult
        qi_scale = block_dict[name + '.qi.scale']
        qo_scale = block_dict[name + '.qo.scale']
        mult, shift = approximate_float(qi_scale / qo_scale)
        param_list.extend([mult, shift])

        # get output section
        param_list.append(('&' + output_section))

        # func call
        func_name = 'arm_'
        if (is_avg):
            func_name = func_name + 'avgpool_s8__with_quantization'
        else:
            func_name = func_name + 'maxpool_s8__with_quantization'

        self.file_writer.write_func_call(func_name, param_list)


    def delpoy_norm(self, name, block_dict, dim_b, dim_c, section):
        # init
        param_list = ['&ctx', '&layernorm_params', '&t_quant_params_1', dim_b, dim_c]

        # weight operation
        weight = block_dict[name + '.layernorm_module.weight']
        weight_name = 'weight_' + self.counter
        param_list.append(('&' + weight_name))
        self.file_writer.write_tensor(weight, weight_name, True, 'q7_t')
        
        # bias operation
        bias = block_dict[name + '.layernorm_module.bias']
        bias_name = 'bias_' + self.counter
        param_list.append(('&' + bias_name))
        self.file_writer.write_tensor(bias, bias_name, True, 'q7_t')

        # output operation and parse conv params
        self.norm_dict['activation.max'] = block_dict[name + '.qw.max']
        self.norm_dict['activation.min'] = block_dict[name + '.qw.min']
        # notice the input offset is negative zero point
        self.norm_dict['input_offset'] = -block_dict[name + '.qi.zero_point']
        self.norm_dict['output_offset'] = block_dict[name + '.qo.zero_point']

        # parse quant params
        qi_scale = block_dict[name + '.qi.scale']
        qo_scale = block_dict[name + '.qo.scale']
        qw_scale = block_dict[name + '.qw.scale']
        M = qi_scale * qw_scale / qo_scale
        mult, shift = approximate_float(M)
        self.t_quant_dict['multiplier'] = mult
        self.t_quant_dict['shift'] = shift
        self.file_writer.write_param_parser('t_quant_params_1', self.t_quant_dict)

        param_list.append('&' + section)
        self.file_writer.write_func_call('arm_layernorm_s8', param_list)

        self.counter += 1


    def deploy_linear(self, name, block_dict, is_sparse,
                    input_section, output_section):
        # init
        param_list = ['&ctx', '&fc_params', '&t_quant_params_1', '&input_dims', ('&' + input_section)]
        
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
        self.fc_dict['activation.max'] = block_dict[name + '.qw.max']
        self.fc_dict['activation.min'] = block_dict[name + '.qw.min']
        # notice the input offset is negative zero point
        self.fc_dict['input_offset'] = -block_dict[name + '.qi.zero_point']
        self.fc_dict['output_offset'] = block_dict[name + '.qo.zero_point']
        self.file_writer.write_param_parser('fc_params', self.fc_dict)

        # parse quant params
        M = block_dict[name + 'M']
        mult, shift = approximate_float(M)

        quant_dict = {}
        quant_dict['multiplier'] = mult
        quant_dict['shift'] = shift
        self.file_writer.write_param_parser('t_quant_params_1', quant_dict)

        if (is_sparse):
            param_list.append(weight.size)

        # function call generate
        func_name = 'arm_fully_connected_s8'
        if (is_sparse):
            func_name = func_name + '_sparse'
        self.file_writer.write_func_call(func_name, param_list)

        self.counter += 1


    def deploy_tokenizer(self, batch, block_dict, inp, oup, input_size,
        origin_section, temp_section):
        # qconv_relu
        self.input_dict['n'] = batch
        self.input_dict['c'], self.output_dict['c'] = inp, oup
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.filter_dict['h'], self.filter_dict['w'] = 3, 3
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 1, 1

        self.deploy_conv(self, 'qconv_relu', block_dict, False, False,
                    origin_section, temp_section)

        # max_pool
        self.input_dict['c'], self.output_dict['c'] = oup, oup
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.filter_dict['h'], self.filter_dict['w'] = 3, 3
        self.output_dict['h'], self.output_dict['w'] = input_size/2, input_size/2
        self.pooling_dict['stride.h'] , self.pooling_dict['stride.w'] = 2, 2
        self.pooling_dict['padding.h'] , self.pooling_dict['padding.w'] = 1, 1

        self.deploy_pooling(self, 'qmaxpool', block_dict, False,
            temp_section, origin_section)


    def deploy_mv2block(self, name, block_dict, batch, inp, oup, input_size,
        input_section, output_section):

        assert(inp == oup)
        # config_list: expansion, stride, is_sparse, output_size
        # name: {*}mv2block{type}_{index} in mv2block
        # {*}mv2block_{index} in downsample
        type_name = name.split('_')[-2]
        if (type_name.endswith('0')):
            config_list = [2, 1, True, input_size]
        elif (type_name.endswith('1')):
            config_list = [3, 1, True, input_size]
        elif (type_name.endswith('2')):
            config_list = [4, 1, True, input_size]
        elif (type_name.endswith('block')):
            config_list = [2, 2, False, input_size/2]
        else:
            print('Cannot parse block as mv2block: ' + name)
            return

        hidden_dim = inp * config_list[0]

        # conv_0
        self.input_dict['n'] = batch
        self.input_dict['c'], self.output_dict['c'] = inp, hidden_dim
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.filter_dict['h'], self.filter_dict['w'] = 1, 1
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 0, 0

        self.deploy_conv(self, 'qconv.0', block_dict, config_list[2], False,
                    input_section, output_section)

        # conv_1
        self.input_dict['n'] = batch
        self.input_dict['c'], self.output_dict['c'] = hidden_dim, hidden_dim
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.filter_dict['h'], self.filter_dict['w'] = 3, 3
        self.output_dict['h'], self.output_dict['w'] = config_list[3], config_list[3]
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = config_list[1], config_list[1]
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 1, 1

        self.deploy_conv(self, 'qconv.1', block_dict, config_list[2], True,
                    output_section, input_section)

        # conv_2
        self.input_dict['n'] = batch
        self.input_dict['c'], self.output_dict['c'] = hidden_dim, oup
        self.input_dict['h'], self.input_dict['w'] = config_list[3], config_list[3]
        self.filter_dict['h'], self.filter_dict['w'] = 1, 1
        self.output_dict['h'], self.output_dict['w'] = config_list[3], config_list[3]
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 0, 0

        self.deploy_conv(self, 'qconv.2', block_dict, config_list[2], False,
                    input_section, output_section)

        print('Block:' + name + ' deploy completed')


    def deploy_transformer(self, name, block_dict, batch, inp, oup, embedding_dim, 
            dim_feedforward, input_size, input_section, output_section, shortcut_section):
        # name: {*}transformer{type}_{index}
        type_name = name.split('_')[-2]
        if (type_name.endswith('0')):
            head_nums = 2
        elif (type_name.endswith('1')):
            head_nums = 4
        else:
            print('Cannot parse block as transformer: ' + name)
            return

        # conv_1: in->out
        self.input_dict['n'] = batch
        self.input_dict['c'], self.output_dict['c'] = inp, inp
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.filter_dict['h'], self.filter_dict['w'] = 3, 3
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 1, 1

        self.deploy_conv('qconv1', block_dict, True, False,
                    input_section, output_section)

        # conv_2: out->in
        self.input_dict['n'] = batch
        self.input_dict['c'], self.output_dict['c'] = inp, embedding_dim
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.filter_dict['h'], self.filter_dict['w'] = 1, 1
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 0, 0

        self.deploy_conv('qconv2', block_dict, True, False,
                    output_section, input_section,)

        # copy in to shortcut, sizeof(q7_t) == 1
        norm_batch = batch * input_size * input_size
        size = norm_batch * embedding_dim * 1
        self.file_writer.writeln('memcpy(&' + shortcut_section
                + ',&' + input_section + ',' + str(size) + ');')

        # pre_norm: in -> in 
        self.delpoy_norm('qpre_norm', block_dict, 
                    norm_batch, embedding_dim, input_section)

        # self_attn: in -> out
        self.input_dict['n'] = batch
        self.input_dict['w'] = input_size * input_size
        self.input_dict['c'] = embedding_dim
        self.deploy_self_atten('self_attn', block_dict, head_nums,
                    input_section, output_section)
        
        # add1: out + shorcut -> in
        self.deploy_add('qadd1', block_dict, output_section, 
            shortcut_section, input_section, size)

        # norm1: in -> in
        self.delpoy_norm('qnorm1', block_dict,
                    norm_batch, embedding_dim, input_section)

        # copy in to shortcut, sizeof(q7_t) == 1
        self.file_writer.writeln('memcpy(&' + shortcut_section
                + ',&' + input_section + ',' + str(size) + ');')

        # linear_relu1: in -> out
        self.input_dict['n'] = norm_batch
        self.filter_dict['n'] = embedding_dim
        self.output_dict['c'] = dim_feedforward
        self.deploy_linear('qlinear_relu1', block_dict, True,
                        input_section, output_section)

        # linear2: out -> in
        self.input_dict['n'] = norm_batch
        self.filter_dict['n'] = dim_feedforward
        self.output_dict['c'] = embedding_dim
        self.deploy_linear('qlinear2', block_dict, True,
                        output_section, input_section)

        # add2: in + shorcut -> out
        self.deploy_add('qadd2', block_dict, input_section, 
            shortcut_section, output_section, size)

        # conv_3: out->in
        self.input_dict['n'] = batch
        self.input_dict['c'], self.output_dict['c'] = embedding_dim, embedding_dim
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.filter_dict['h'], self.filter_dict['w'] = 3, 3
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 1, 1

        self.deploy_conv('qconv3', block_dict, True, False,
                    input_section, output_section)

        # conv_4: in->out
        self.input_dict['n'] = batch
        self.input_dict['c'], self.output_dict['c'] = embedding_dim, oup
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.filter_dict['h'], self.filter_dict['w'] = 1, 1
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 0, 0

        self.deploy_conv('qconv4', block_dict, True, False,
                    output_section, input_section)
        
        print('Block:' + name + ' deploy completed')


    def deploy_last_conv(self, name, block_dict, batch, inp, oup, input_size,
            input_section, output_section):

        # just one conv
        self.input_dict['n'] = batch
        self.input_dict['c'], self.output_dict['c'] = inp, oup
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.filter_dict['h'], self.filter_dict['w'] = 1, 1
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 0, 0

        self.deploy_conv('', block_dict, True, False,
                    input_section, output_section)

        print('Block:' + name + ' deploy completed')


    def deploy_global_pooling(self, name, block_dict, batch, inp, oup, input_size,
            input_section, output_section):

        if (name == 'qglobal_pooling'):
            # todo: qglobal_pooling
            pass
        elif (name == 'global_pooling'):
            # global_pooling, avgpooling
            self.input_dict['c'], self.output_dict['c'] = inp, oup
            self.input_dict['h'], self.input_dict['w'] = input_size, input_size
            self.filter_dict['h'], self.filter_dict['w'] = input_size, input_size
            self.output_dict['h'], self.output_dict['w'] = 1, 1
            self.pooling_dict['stride.h'] , self.pooling_dict['stride.w'] = input_size, input_size
            self.pooling_dict['padding.h'] , self.pooling_dict['padding.w'] = 0, 0

            self.deploy_pooling(self, '', block_dict, True,
                input_section, output_section)
        else: 
            print('Cannot parse block as global_pooling: ' + name)
            return

        print('Block:' + name + ' deploy completed')


    def deploy_classifier(self, name, block_dict, batch, inp, oup, input_size,
            input_section, output_section):

        # just one linear
        self.input_dict['n'] = batch
        self.filter_dict['n'] = inp
        self.output_dict['c'] = oup
        self.deploy_linear('', block_dict, True,
                        input_section, output_section)

        print('Block:' + name + ' deploy completed')


    def deploy_model(self):
        model_config_path = self.config['model_config_path']
        with open(model_config_path, 'r') as file:
            model_config = yaml.load(file, Loader=yaml.FullLoader)
        print(model_config)


    def print_tensor(self):
        for key, value in self.downsample_list_dict.items():
            print("\nblock: " + key)
            for key0, value0 in value.items():
                print(("key: " + key0).ljust(40, ' '), end="")
                if (type(value0) == float):
                    print(value0)
                else:
                    print(value0.shape)
        for key, value in self.mv2block_list_dict.items():
            print("\nblock: " + key)
            for key0, value0 in value.items():
                print(("key: " + key0).ljust(40, ' '), end="")
                if (type(value0) == float):
                    print(value0)
                else:
                    print(value0.shape)
        for key, value in self.transformer_list_dict.items():
            print("\nblock: " + key)
            for key0, value0 in value.items():
                print(("key: " + key0).ljust(40, ' '), end="")
                if (type(value0) == float):
                    print(value0)
                else:
                    print(value0.shape)
        print("\nblock: last_conv")
        for key, value in self.last_conv_dict.items():
            print(("key: " + key).ljust(40, ' '), end="")
            if (type(value) == float):
                print(value)
            else:
                print(value.shape)
        print("\nblock: pooling")
        for key, value in self.pooling_dict.items():
            print(("key: " + key).ljust(40, ' '), end="")
            if (type(value) == float):
                print(value)
            else:
                print(value.shape)
        print("\nblock: classifier")
        for key, value in self.classifier_dict.items():
            print(("key: " + key).ljust(40, ' '), end="")
            if (type(value) == float):
                print(value)
            else:
                print(value.shape)


if __name__ == '__main__':
    print("start")
    model_deployment = Model_deployment()
    model_deployment.load_tensor()
    model_deployment.print_tensor()
    model_deployment.deploy_model()