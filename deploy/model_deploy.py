import os
import numpy as np
import yaml
from utils import conv_data_to_sparse, approximate_float
from file_write import File_writer

class Model_deployment():
    def __init__(self):

        # basic info
        with open('config.yml', 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        self.tensor_path = '../TinySPOS/' + self.config['tensor_path']
        self.file_writer = File_writer(self.config['func_path'], self.config['data_path'])
        with open('../TinySPOS/' + self.config['memory_list'], 'r') as file:
            self.size_list = yaml.load(file, Loader=yaml.FullLoader)
        self.max_size = self.config['ram_size'] * 1024  # kb
        self.ori_max_size = self.max_size

        # dict list
        self.downsample_list_dict = {}
        self.mv2block_list_dict = {}
        self.transformer_list_dict = {}
        self.last_conv_dict = {}
        self.pooling_list_dict = {}
        self.classifier_dict = {}
        
        # const dict
        self.input_dict = {}        # input_dims
        self.filter_dict = {}       # filter_dims
        self.output_dict = {}       # filter_dims
        self.conv_dict = {}         # conv_params
        self.fc_dict = {}           # fc_params
        self.norm_dict = {}         # norm_params
        self.pooling_dict = {}      # pool params
        self.t_quant_dict = {}      # per_tensor
        self.c_quant_dict = {}      # per_channel

        # block counter for conv and linear
        self.counter = 0
        self.first = 1

        self.image = np.load(self.config['image_path']).transpose(0, 2, 3, 1)
        self.image_class = self.config['image_class']
        self.size = self.image.shape[1]
    

    def get_size_output(self):
        cur_layer_size = self.size_list.pop(0)
        sum = 0
        for item in cur_layer_size:
            sum += item
        
        if (sum > self.ori_max_size):
            print('Warning! The max ram usage ' + str(sum) + ' may be out of the ' 
                + str(self.ori_max_size) + ' bound!')
        return cur_layer_size[0], cur_layer_size[1]


    def load_tensor(self):

        tensor_dir = os.listdir(self.tensor_path)
        
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
                if (name_list[0] in self.pooling_list_dict.keys()):
                    block_dict = self.pooling_list_dict[name_list[0]]
                else:
                    block_dict = {}
                    self.pooling_list_dict[name_list[0]] = block_dict
            elif (name_list[0] == "qclassifier"):
                block_dict = self.classifier_dict
            else:
                print("Cannot resolve " + file)
                continue

            # keyname: {block_type}.{block_name}
            if (name_list[0] == "downsample" or name_list[0] == "mv2block"
                or name_list[0] == "transformer"):
                key_name = name_list[2]
                for i in range(3, len(name_list) - 1):
                    key_name = key_name + '.' + name_list[i]
                value = np.load(self.tensor_path + "/" + file)
                if (value.size == 1):
                    value = float(value)
                block_dict[key_name] = value
            else:
                key_name = name_list[1]
                for i in range(2, len(name_list) - 1):
                    key_name = key_name + '.' + name_list[i]
                value = np.load(self.tensor_path + "/" + file)
                if (value.size == 1):
                    value = float(value)
                block_dict[key_name] = value
        
        # sort by the index: {blockname}_{index}
        self.downsample_list_dict = dict(sorted(self.downsample_list_dict.items(), 
                key=lambda x: int(x[0].split('_')[-1])))
        self.mv2block_list_dict = dict(sorted(self.mv2block_list_dict.items(), 
                key=lambda x: int(x[0].split('_')[-1])))
        self.transformer_list_dict = dict(sorted(self.transformer_list_dict.items(), 
                key=lambda x: int(x[0].split('_')[-1])))


    def deploy_add(self, name:str, block_dict:dict, 
            in_section_1:str, in_section_2:str, out_section:str, count:int):
        
        # init
        param_list = [in_section_1, in_section_2]

        # quant param
        M1 = block_dict[name + 'M1']
        M2 = block_dict[name + 'M2']
        qi1_mult, qi1_shift = approximate_float(M1)
        qi2_mult, qi2_shift = approximate_float(M2)

        qi1_offset = -block_dict[name + 'qi1.zero_point']
        qi2_offset = -block_dict[name + 'qi2.zero_point']
        qo_offset = block_dict[name + 'qo.zero_point']
        
        # func call
        param_list.extend([qi1_offset, qi1_mult, qi1_shift, qi2_offset, qi2_mult, qi2_shift,
            0, out_section, qo_offset, 0x7fffffff, 0, -128, 127, count])
        self.file_writer.write_func_call('arm_elementwise_add_s8_with_neg', param_list)


    def deploy_conv(self, name:str, block_dict:dict, is_sparse:bool, is_depthwise:bool, 
            in_section:str, out_section:str):

        # init
        param_list = ['&ctx']
        if (is_depthwise):
            param_list.append('&dw_conv_params')
        else:
            param_list.append('&conv_params')
        param_list.extend(['&c_quant_params', '&input_dims', in_section])

        # weight operation
        weight = block_dict[name + 'conv_module.weight']
        # channel & filter dim info

        # conv: (out_ch, in_ch, h, w)->(out_ch, h, w, in_ch)
        # transpose: (0, 2, 3, 1)
        # out ch = out_ch(ori dim 0), in ch = in_ch(ori dim 1)

        # dw conv without sparse: (out_ch(in_ch), 1, h, w)->(1, h, w, out_ch(in_ch))
        # transpose: (1, 2, 3, 0)
        # out ch = out_ch(ori dim 0), in ch = out_ch(ori dim 0)

        # dw conv with sparse: (out_ch(in_ch), 1, h, w)->(out_ch(in_ch), h, w, 1)
        # transpose: (0, 2, 3, 1)
        # out ch = out_ch(ori dim 0), in ch = 1(ori dim 1)

        ori_shape = weight.shape
        self.output_dict['c'] = ori_shape[0]
        if (is_depthwise and not is_sparse):
            self.input_dict['c'] = ori_shape[0]
        else:
            self.input_dict['c'] = ori_shape[1]
        self.filter_dict['h'], self.filter_dict['w'] = ori_shape[2], ori_shape[3]

        # sparse encode
        if (is_depthwise and not is_sparse):
            weight = weight.transpose((1, 2, 3, 0))
        else:
            weight = weight.transpose((0, 2, 3, 1))

        if (is_sparse):
            weight, is_sparse = conv_data_to_sparse(weight)
            # this branch goes to dwconv:
            # should be sparse, but after sparse encode the tensor got larger
            if (is_depthwise and not is_sparse):
                weight = weight.transpose((1, 2, 3, 0))
                self.input_dict['c'] = ori_shape[0]

        self.file_writer.write_param_parser('input_dims', self.input_dict)
        self.file_writer.write_param_parser('filter_dims', self.filter_dict)
        self.file_writer.write_param_parser('output_dims', self.output_dict)

        weight_name = 'weight_' + str(self.counter)
        param_list.extend(['&filter_dims', weight_name])
        self.file_writer.write_const_tensor(weight, weight_name, 'q7_t')
        
        # bias operation
        if ((name + 'conv_module.bias') in block_dict.keys()):
            bias = block_dict[name + 'conv_module.bias']
            bias_name = 'bias_' + str(self.counter)
            param_list.extend(['&bias_dims', bias_name])
            self.file_writer.write_const_tensor(bias, bias_name, 'q31_t')
        else:
            param_list.extend(['&bias_dims', 'NULL'])
        

        # output operation and parse conv params
        param_list.extend(['&output_dims', out_section])
        self.conv_dict['activation.max'] = 127
        self.conv_dict['activation.min'] = -128
        # notice the input offset is negative zero point
        self.conv_dict['input_offset'] = -block_dict[name + 'qi.zero_point']
        self.conv_dict['output_offset'] = block_dict[name + 'qo.zero_point']
        # if first conv, transform from int8 to uint
        if (self.first == 1):
            self.conv_dict['input_offset'] += 128
            self.first = 0
        
        # dilation
        self.conv_dict['dilation.h'] = 1
        self.conv_dict['dilation.w'] = 1

        if (is_depthwise):
            self.conv_dict['ch_mult'] = 1
            self.file_writer.write_param_parser('dw_conv_params', self.conv_dict)
            self.conv_dict.pop('ch_mult')
        else:
            self.file_writer.write_param_parser('conv_params', self.conv_dict)

        # parse quant params
        M = block_dict[name + 'M']
        mult, shift = approximate_float(M)
        mult_name = 'mult_' + str(self.counter)
        shift_name = 'shift_' + str(self.counter)
        self.file_writer.write_const_tensor(mult, mult_name, 'q31_t')
        self.file_writer.write_const_tensor(shift, shift_name, 'q31_t')
        
        self.file_writer.writeln('memcpy(conv_mult_use,' + mult_name + ',' + str(4*mult.size) + ');','func')
        self.file_writer.writeln('memcpy(conv_shift_use,' + shift_name + ',' + str(4*shift.size) + ');','func')

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


    def deploy_linear(self, name:str, block_dict:dict, is_sparse:bool,
            in_section:str, out_section:str):

        # init
        param_list = ['&ctx', '&fc_params', '&t_quant_params', '&input_dims', 
            in_section]
        
        # weight operation
        weight = block_dict[name + 'fc_module.weight']
        weight_shape = weight.shape
        if (is_sparse):
            weight, is_sparse = conv_data_to_sparse(weight)
        weight_name = 'weight_' + str(self.counter)
        param_list.extend(['&filter_dims', weight_name])
        self.file_writer.write_const_tensor(weight, weight_name, 'q7_t')
        
        self.filter_dict['n'] = weight_shape[1]
        self.output_dict['c'] = weight_shape[0]
        
        # bias operation
        if ((name + 'fc_module.bias') in block_dict.keys()):
            bias = block_dict[name + 'fc_module.bias']
            if (type(bias) == float):
                bias = np.array([int(bias)])
            bias_name = 'bias_' + str(self.counter)
            param_list.extend(['&bias_dims', bias_name])
            self.file_writer.write_const_tensor(bias, bias_name, 'q31_t')
        else:
            param_list.extend(['&bias_dims', 'NULL'])
        

        # output operation and parsefc params
        param_list.extend(['&output_dims', out_section])
        self.fc_dict['activation.max'] = 127
        self.fc_dict['activation.min'] = -128
        # notice the input offset is negative zero point
        self.fc_dict['input_offset'] = -block_dict[name + 'qi.zero_point']
        self.fc_dict['output_offset'] = block_dict[name + 'qo.zero_point']
        self.file_writer.write_param_parser('fc_params', self.fc_dict)

        # parse quant params
        M = block_dict[name + 'M']
        mult, shift = approximate_float(M)

        quant_dict = {}
        quant_dict['multiplier'] = mult
        quant_dict['shift'] = shift
        self.file_writer.write_param_parser('t_quant_params', quant_dict)
        self.file_writer.write_param_parser('input_dims', self.input_dict)
        self.file_writer.write_param_parser('filter_dims', self.filter_dict)
        self.file_writer.write_param_parser('output_dims', self.output_dict)

        if (is_sparse):
            param_list.append(weight.size)
            
        # function call generate
        func_name = 'arm_fully_connected_s8'
        if (is_sparse):
            func_name = func_name + '_sparse'
        self.file_writer.write_func_call(func_name, param_list)

        self.counter += 1


    def deploy_pooling(self, name:str, block_dict:dict, is_avg:bool,
            in_section:str, out_section:str):

        # init
        param_list = ['&ctx', '&pool_params', '&input_dims', in_section, 
                '&filter_dims', '&output_dims']
        
        self.file_writer.write_param_parser('input_dims', self.input_dict)
        self.file_writer.write_param_parser('filter_dims', self.filter_dict)
        self.file_writer.write_param_parser('output_dims', self.output_dict)
        self.file_writer.write_param_parser('pool_params', self.pooling_dict)

        # get offset
        param_list.append(-block_dict[name + 'qi.zero_point'])
        param_list.append(block_dict[name + 'qo.zero_point'])

        # get input mult
        qi_scale = block_dict[name + 'qi.scale']
        qo_scale = block_dict[name + 'qo.scale']
        mult, shift = approximate_float(qi_scale / qo_scale)
        param_list.extend([mult, shift, out_section])

        # func call
        func_name = 'arm_'
        if (is_avg):
            func_name = func_name + 'avgpool_s8_with_quantization'
        else:
            func_name = func_name + 'maxpool_s8_with_quantization'

        self.file_writer.write_func_call(func_name, param_list)


    def deploy_matmul(self, name:str, block_dict, dim_b:int, dim_lr:int, dim_lc:int,
            dim_rc:int, is_trans:bool, in_section_1:str, in_section_2:str, out_section:str):
        
        # init
        param_list = [in_section_1, in_section_2, 'NULL', out_section]
        if (not is_trans):
            param_list.insert(0, '&ctx')
            
        # output quant
        M = block_dict[name + 'M']
        qo_mult, qo_shift = approximate_float(M)
        param_list.extend([qo_mult, qo_shift])

        # dim info
        param_list.extend([dim_lr, dim_lc, dim_rc])

        # offset
        qi1_offset = -block_dict[name + 'qi1.zero_point']
        qi2_offset = -block_dict[name + 'qi2.zero_point']
        qo_offset = block_dict[name + 'qo.zero_point']
        param_list.extend([qi1_offset, qi2_offset, qo_offset, dim_b, -128, 127])
        
        if (is_trans):
            func_name = 'arm_nn_batch_mat_mult_nt_t_s8'
        else:
            func_name = 'arm_nn_batch_mat_mult_s8'
        self.file_writer.write_func_call(func_name, param_list)


    def deploy_softmax(self, name:str, block_dict, dim_b:int, dim_c:int,
            in_section:str, out_section:str):

        # init
        param_list = [in_section, dim_b, dim_c]

        # input quant
        qi_scale = block_dict[name + 'qi.scale']
        softmax_input_integer_bits = 5
        qi_scale = min(qi_scale * (1 << (31 - softmax_input_integer_bits)),
                                   (1 << 31) - 1)
        qi_mult, qi_shift = approximate_float(qi_scale)

        param_list.extend([qi_mult, qi_shift, -248, 
                out_section])
        self.file_writer.write_func_call('arm_softmax_s8', param_list)


    def deploy_self_attn(self, name:str, block_dict:dict,
            b:int, n:int, c:int, head_nums:int, section:str):
        
        # self_attention start
        # qkv linear: head -> tail
        # from (b n c) to (b n 3c)
        _, bncx3_size = self.get_size_output()
        self.input_dict['n'] = b * n
        self.deploy_linear(name + 'qqkv.', block_dict, True,
                    section, '&'+section+'['+str(self.max_size-bncx3_size)+']')
        
        # bnc transpose: tail -> head
        # from (b, n, 3, head_nums, c / head_nums) 
        # to (3, head_nums, b, n, c / head_nums)
        # n = input_size * input_size (image size), c = embedding_dim (channel)
        # current memory: [..... q k v | reserve]
        self.get_size_output()
        self.deploy_transpose((b * n), (3 * head_nums),
                    (c / head_nums), '&'+section+'['+str(self.max_size-bncx3_size)+']',
                    section)
        self.file_writer.writeln('memcpy(&' + section + '['  + str(self.max_size-bncx3_size)
                    + '],' + section + ',' + str(bncx3_size) + ');', 'func')

        # q * k^T
        # from (head_nums*b, n, c / head_nums) * (head_nums*b, c / head_nums, n)
        # to (head_nums*b, n, n)
        # current memory: [q*k ... q k v | reserve]
        _, attn_size = self.get_size_output()
        self.deploy_matmul('self_attn.qmatmul_qk.', block_dict, (head_nums * b),
                    n, (c / head_nums),
                    n, True,
                    '&' + section + '[' + str(self.max_size - bncx3_size)+ ']',  # q
                    '&' + section + '[' + str(self.max_size - bncx3_size*2//3)+ ']',  # k
                    section)

        # in_place softmax
        self.get_size_output()
        self.deploy_softmax('self_attn.qsoftmax1.', block_dict, 
                    head_nums * b * n, 
                    n, 
                    section, section)

        # attn * v
        # from (head_nums*b, n, n) * (head_nums*b, n, c / head_nums)
        # to (head_nums*b, n, c / head_nums)
        # current memory: [attn attn*v ... v | reserve]
        self.get_size_output()
        self.deploy_matmul('self_attn.qmatmul_attnv.', block_dict, (head_nums * b),
                    n, n, (c / head_nums), False,
                    section, # attn
                    '&' + section + '[' + str(self.max_size - bncx3_size//3)+ ']',  # v
                    '&' + section + '[' + str(attn_size)+ ']')

        # bnc transpose
        # from (head_nums*b, n, c / head_nums)
        # to (b, n, c)
        # current memory: [(attn) attn*v ... bnc_value | reserve]
        _, bnc_size = self.get_size_output()
        self.deploy_transpose(head_nums, (b * n),
                    (c / head_nums), '&' + section + '[' + str(attn_size)+ ']',
                    '&' + section + '[' + str(self.max_size - bnc_size) + ']')

        # self attention final linear
        # current memory: [proj ... bnc_value | reserve]
        self.get_size_output()
        self.input_dict['n'] = b * n
        self.deploy_linear('self_attn.qproj.', block_dict, True,
                    '&' + section + '[' + str(self.max_size - bnc_size) + ']', section)


    def deploy_norm(self, name:str, block_dict:dict, dim_b:int, dim_c:int, 
            in_section:str, out_section:str):

        # init
        param_list = ['&ctx', '&norm_params', '&t_quant_params', dim_b, dim_c]

        # output operation and parse conv params
        self.norm_dict['activation.max'] = 127
        self.norm_dict['activation.min'] = -128
        # notice the input offset is negative zero point
        self.norm_dict['input_offset'] = -block_dict[name + 'qi.zero_point']
        self.norm_dict['output_offset'] = block_dict[name + 'qo.zero_point']
        self.file_writer.write_param_parser('norm_params', self.norm_dict)

        # weight operation
        weight = block_dict[name + 'layernorm_module.weight']
        weight_name = 'weight_' + str(self.counter)
        param_list.append(weight_name)
        self.file_writer.write_const_tensor(weight, weight_name, 'q7_t')
        
        # bias operation
        bias = block_dict[name + 'layernorm_module.bias']
        bias_name = 'bias_' + str(self.counter)
        param_list.append(bias_name)
        self.file_writer.write_const_tensor(bias, bias_name, 'q31_t')

        # parse quant params
        M = block_dict[name + 'M']
        mult, shift = approximate_float(M)
        self.t_quant_dict['multiplier'] = mult
        self.t_quant_dict['shift'] = shift
        self.file_writer.write_param_parser('t_quant_params', self.t_quant_dict)

        param_list.extend([in_section, out_section])
        self.file_writer.write_func_call('arm_nn_layernorm_s8', param_list)

        self.counter += 1


    def deploy_transpose(self, dim_b:int, dim_n:int, dim_c:int,
            input_section:str, output_section:str):

        param_list = [dim_b, dim_n, dim_c, input_section, output_section]
        self.file_writer.write_func_call('arm_nn_transpose_bnc_to_nbc_q7', param_list)


    def deploy_tokenizer(self, batch:int, block_dict:dict,
            input_size:int, section:str):
        # qconv_relu: head -> tail
        _, size_0 = self.get_size_output()
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 1, 1
        self.deploy_conv(self, 'qconv_relu.', block_dict, False, False,
                    section, '&'+section+'['+str(self.max_size - size_0[1])+']')

        # max_pool: tail -> head
        self.get_size_output()
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.input_dict['c'] = self.output_dict['c']
        self.filter_dict['h'], self.filter_dict['w'] = 3, 3
        self.output_dict['h'], self.output_dict['w'] = input_size/2, input_size/2
        self.pooling_dict['stride.h'] , self.pooling_dict['stride.w'] = 2, 2
        self.pooling_dict['padding.h'] , self.pooling_dict['padding.w'] = 1, 1
        self.pooling_dict['activation.min'] , self.pooling_dict['activation.max'] = -128, 127

        self.deploy_pooling('qmaxpool.', block_dict, False,
                '&'+section+'['+str(self.max_size - size_0[1])+']', section)


    def deploy_mv2block(self, batch:int, name:str, block_dict:dict, 
            input_size:int, section:str):

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

        res = False
        if (config_list[1] == 1):
            inp = int(block_dict['qconv.0.conv_module.weight'].shape[1])
            oup = int(block_dict['qconv.2.conv_module.weight'].shape[0])
            if (inp == oup):
                res = True
        
        if (res):
            feature_size = int(inp * input_size * input_size)
            self.file_writer.writeln('memcpy(&' + section + '[' + str(self.max_size - feature_size) 
                + '],' + section + ',' + str(feature_size) + ');', 'func')
            self.max_size -= feature_size

        # conv_0: head -> tail
        _, conv0_size = self.get_size_output()
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 0, 0

        self.deploy_conv('qconv.0.', block_dict, config_list[2], False,
                    section, '&'+section+'['+str(self.max_size - conv0_size)+']')

        # conv_1: tail -> head
        self.get_size_output()
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.output_dict['h'], self.output_dict['w'] = config_list[3], config_list[3]
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = config_list[1], config_list[1]
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 1, 1

        self.deploy_conv('qconv.1.', block_dict, config_list[2], True,
                '&'+section+'['+str(self.max_size - conv0_size)+']', section)

        # conv_2: head -> tail
        _, conv2_size = self.get_size_output()
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = config_list[3], config_list[3]
        self.output_dict['h'], self.output_dict['w'] = config_list[3], config_list[3]
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 0, 0

        self.deploy_conv('qconv.2.', block_dict, config_list[2], False,
                    section, '&'+section+'['+str(self.max_size - conv2_size)+']')

        if (res):
            self.get_size_output()
            self.max_size += feature_size
            self.deploy_add('qadd.', block_dict,
                '&'+section+'['+str(self.max_size-feature_size*2)+']', 
                '&'+section+'['+str(self.max_size-feature_size)+']',                       
                section,feature_size)
        else:
            # copy output from tail to head
            self.file_writer.writeln('memcpy(' + section + ',&' + section 
                    + '[' + str(self.max_size - conv2_size)+'],' + str(conv2_size) + ');', 'func')

        print('Block:' + name + ' deploy completed')


    def deploy_transformer(self, batch:int, name:str, block_dict:dict,
            input_size:int, section:str):

        # name: {*}transformer{type}_{index}
        type_name = name.split('_')[-2]
        if (type_name.endswith('0')):
            head_nums = 2
        elif (type_name.endswith('1')):
            head_nums = 4
        else:
            print('Cannot parse block as transformer: ' + name)
            return

        # conv_1: head -> tail
        _, conv1_size = self.get_size_output()
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 1, 1

        self.deploy_conv('qconv1.', block_dict, True, False,
                    section, '&'+section+'['+str(self.max_size - conv1_size)+']')

        # conv_2: tail -> head
        _, bnc_size = self.get_size_output()
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 0, 0

        self.deploy_conv('qconv2.', block_dict, True, False,
                    '&'+section+'['+str(self.max_size - conv1_size)+']', section)

        embedding_dim = self.output_dict['c']

        # copy shortcut data from head to tail
        norm_batch = batch * input_size * input_size
        self.file_writer.writeln('memcpy(&' + section + '[' + str(self.max_size - bnc_size)
                    + '],' + section + ',' + str(bnc_size) + ');', 'func')
        self.max_size -= bnc_size

        # pre_norm: head -> head
        self.get_size_output()
        self.deploy_norm('qpre_norm.', block_dict, norm_batch, embedding_dim,
                    section, section)

        # self attention
        self.deploy_self_attn('self_attn.', block_dict,
            batch, (input_size * input_size), embedding_dim, head_nums, section)
        
        # add1: head + tail -> mid
        self.max_size += bnc_size
        self.get_size_output()
        self.deploy_add('qadd1.', block_dict, section, 
            '&'+section+'['+str(self.max_size-bnc_size)+']',
            '&'+section+'['+str(bnc_size)+']',bnc_size)

        # copy to head
        self.file_writer.writeln('memcpy(' + section + ',&'
                    + section+'['+str(bnc_size)+'],' + str(bnc_size) + ');', 'func')

        # norm1: head -> head
        self.get_size_output()
        self.deploy_norm('qnorm1.', block_dict, norm_batch, 
                    embedding_dim, section, section)

        # copy head to shortcut
        self.file_writer.writeln('memcpy(&' + section + '[' + str(self.max_size-bnc_size) + '],'
                + section + ',' + str(bnc_size) + ');', 'func')
        self.max_size -= bnc_size
        
        # linear relu1: head -> tail
        _, linear_size = self.get_size_output()
        self.input_dict['n'] = norm_batch
        self.deploy_linear('qlinear_relu1.', block_dict, True,
                    section, '&'+section+'['+str(self.max_size-linear_size)+']')

        # linear2: tail -> head
        _, linear2_size = self.get_size_output()
        self.input_dict['n'] = norm_batch
        self.deploy_linear('qlinear2.', block_dict, True,
                    '&'+section+'['+str(self.max_size-linear_size)+']', section)

        # add2: head + tail -> mid
        self.max_size += bnc_size
        self.get_size_output()
        self.deploy_add('qadd2.', block_dict,
            section,
            '&'+section+'['+str(self.max_size-bnc_size)+']', 
            '&'+section+'['+str(linear2_size)+']',linear2_size)

        # copy add2 to head
        self.file_writer.writeln('memcpy(' + section + ',&' + section +
                '[' + str(linear2_size) + '],' + str(linear2_size) + ');', 'func')

        # conv_3: head -> tail
        _, conv3_size = self.get_size_output()
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 1, 1

        self.deploy_conv('qconv3.', block_dict, True, False,
                    section, '&'+section+'['+str(self.max_size-conv3_size)+']')

        # conv_4: tail -> head
        self.get_size_output()
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 0, 0

        self.deploy_conv('qconv4.', block_dict, True, False,
                    '&'+section+'['+str(self.max_size-conv3_size)+']', section)
        
        print('Block:' + name + ' deploy completed')


    def deploy_last_conv(self, batch:int, name:str, block_dict:dict,
            input_size:int, section:str):

        # just one conv, head -> tail
        _, conv_size = self.get_size_output()
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 0, 0

        self.deploy_conv('', block_dict, True, False,
                    section, '&'+section+'['+str(self.max_size - conv_size)+']')

        # copy output from tail to head
        self.file_writer.writeln('memcpy(' + section + ',&' + section + '[' 
                    + str(self.max_size - conv_size)+'],' + str(conv_size) + ');', 'func')

        print('Block:' + name + ' deploy completed')


    def deploy_global_pooling(self, batch:int, name:str, block_dict:dict,
            input_size:int, channel:int, section:str):

        if (name == 'global_pooling'):
            # qglobal_pooling
            # shortcut storage: head -> tail
            in_size, out_size = self.get_size_output()
            self.file_writer.writeln('memcpy(&' + section + '[' + str(self.max_size - in_size)
                + '],' + section + ',' + str(in_size) + ');', 'func')
            self.max_size -= in_size

            # linear: head -> tail
            # shape: b (h w) c -> b (h w) 1 -> b (h w) 1
            self.input_dict['n'] = batch * input_size * input_size
            self.deploy_linear('qattention_pool.', block_dict, False,
                        section, '&'+section+'['+str(self.max_size - out_size)+']')
            
            # qsoftmax: tail -> tail
            self.get_size_output()
            self.deploy_softmax('qsoftmax.', block_dict, batch, (input_size * input_size), 
                        '&'+section+'['+str(self.max_size - out_size)+']', 
                        '&'+section+'['+str(self.max_size - out_size)+']')

            # matmul: tail * tail -> mid
            # shape: b 1 (h w) * b (h w) c -> b 1 c -> b c
            self.get_size_output()
            self.max_size += in_size
            self.deploy_matmul('qmatmul.', block_dict, batch, 1, (input_size * input_size),
                        channel, False, 
                        '&'+section + '[' + str(self.max_size - in_size - out_size)+ ']', 
                        '&'+section + '[' + str(self.max_size - in_size)+ ']', 
                        section)

            
        elif (name == 'qglobal_pooling'):
            # global_pooling, avgpooling: head -> tail
            self.input_dict['n'] = batch
            self.input_dict['h'], self.input_dict['w'] = input_size, input_size
            self.input_dict['c'] = channel
            self.filter_dict['h'], self.filter_dict['w'] = input_size, input_size
            self.output_dict['h'], self.output_dict['w'] = 1, 1
            self.pooling_dict['stride.h'] , self.pooling_dict['stride.w'] = input_size, input_size
            self.pooling_dict['padding.h'] , self.pooling_dict['padding.w'] = 0, 0
            self.pooling_dict['activation.min'] , self.pooling_dict['activation.max'] = -128, 127

            pool_size = batch * channel
            self.deploy_pooling('', block_dict, True,
                section, '&'+section+'['+str(self.max_size - pool_size)+']')
            
            # copy from tail to head
            self.file_writer.writeln('memcpy(' + section + ',&' + section 
                +'['+str(self.max_size-pool_size)+'],' + str(pool_size) + ');', 'func')

        else: 
            print('Cannot parse block as global_pooling: ' + name)
            return

        print('Block:' + name + ' deploy completed')


    def deploy_classifier(self, batch:int, name:str, block_dict:dict,
            section:str):

        # just one linear
        _, linear_size = self.get_size_output()
        self.input_dict['n'] = batch
        self.deploy_linear('', block_dict, True,
                        section, '&'+section+'['+str(self.max_size-linear_size)+']')

        # copy from tail to head
        self.file_writer.writeln('memcpy(' + section + ',&' + section 
            +'['+str(self.max_size-linear_size)+'],' + str(linear_size) + ');', 'func')

        print('Block:' + name + ' deploy completed')


    def deploy_model(self, batch=1):
        model_config_path = '../TinySPOS/configs/' + self.config['model_config_path'] + \
                            '/quant_dmtp_single.yml'
        with open(model_config_path, 'r') as file:
            model_config = yaml.load(file, Loader=yaml.FullLoader)
        

        last_channel = model_config['last_channel']
        section = 'section'

        # init file
        with open(self.config['func_init'], 'r') as r_file:
            self.file_writer.write_file(r_file, 'func')

        with open(self.config['data_init'], 'r') as r_file:
            self.file_writer.write_file(r_file, 'data')

        self.file_writer.writeln('static int32_t conv_mult_use[' + str(last_channel) + ']={0};', 'func')
        self.file_writer.writeln('static int32_t conv_shift_use[' + str(last_channel) + ']={0};', 'func')
        self.file_writer.writeln('c_quant_params.multiplier=conv_mult_use;', 'func')
        self.file_writer.writeln('c_quant_params.shift=conv_shift_use;', 'func')

        self.file_writer.writeln('static q7_t ' + section + '[' + str(self.max_size) + ']={0};', 'func')
        self.file_writer.write_const_tensor(self.image, 'image', 'q7_t')
        self.file_writer.writeln('memcpy(&section,&image,3072);', 'func')
        
        # deploy quantization inference
        for key, value in self.downsample_list_dict.items():
            self.file_writer.writeln('// block: ' + key, 'func')
            key_list = key.split('_')
            if (key_list[1] == 'tokenizer'):
                self.deploy_tokenizer(batch, value, self.size, section)
            elif (key_list[1] == 'mv2block'):
                self.deploy_mv2block(batch, key, value, self.size, section)
            self.size /= 2
            print('-'*70)

        for key, value in self.mv2block_list_dict.items():
            self.file_writer.writeln('// block: ' + key, 'func')
            self.deploy_mv2block(batch, key, value, self.size, section)
            print('-'*70)

        embedding_dim = []
        embedding_dim.append(model_config['transformer0_embedding_dim'])
        embedding_dim.append(model_config['transformer1_embedding_dim'])

        for key, value in self.transformer_list_dict.items():
            self.file_writer.writeln('// block: ' + key, 'func')
            self.deploy_transformer(batch, key, value, self.size, section)
            print('-'*70)

        self.file_writer.writeln('// block: last conv', 'func')
        self.deploy_last_conv(batch, 'last_conv', self.last_conv_dict,
            self.size, section)
        print('-'*70)

        for key, value in self.pooling_list_dict.items():
            self.file_writer.writeln('// block: ' + key, 'func')
            self.deploy_global_pooling(batch, key, value,
                self.size, last_channel, section)
            print('-'*70)

        self.file_writer.writeln('// block: classifier', 'func')
        self.deploy_classifier(batch, 'classifier', self.classifier_dict,
            section)
    
        self.file_writer.writeln('result_check(&ctx,section,' + str(self.image_class) + ');', 'func')
        self.file_writer.writeln('return 0;\n}\n', 'func')
        self.file_writer.writeln('\n#endif', 'data')

        print('Model deploy completed')
        if (len(self.size_list) != 0):
            print('Warning! Size list may not be set properly!')
            print('Remaining size list count: ' + str(len(self.size_list)) + ' (0 is correct)')


    def print_tensor(self):
        for key, value in self.downsample_list_dict.items():
            print("\nblock: " + key)
            for key0, value0 in value.items():
                print(("key: " + key0).ljust(70, ' '), end="")
                if (type(value0) == float):
                    print(value0)
                else:
                    print(value0.shape)
        for key, value in self.mv2block_list_dict.items():
            print("\nblock: " + key)
            for key0, value0 in value.items():
                print(("key: " + key0).ljust(70, ' '), end="")
                if (type(value0) == float):
                    print(value0)
                else:
                    print(value0.shape)
        for key, value in self.transformer_list_dict.items():
            print("\nblock: " + key)
            for key0, value0 in value.items():
                print(("key: " + key0).ljust(70, ' '), end="")
                if (type(value0) == float):
                    print(value0)
                else:
                    print(value0.shape)
        print("\nblock: qlast_conv")
        for key, value in self.last_conv_dict.items():
            print(("key: " + key).ljust(70, ' '), end="")
            if (type(value) == float):
                print(value)
            else:
                print(value.shape)
        for key, value in self.pooling_list_dict.items():
            print("\nblock: " + key)
            for key0, value0 in value.items():
                print(("key: " + key0).ljust(70, ' '), end="")
                if (type(value0) == float):
                    print(value0)
                else:
                    print(value0.shape)
        print("\nblock: qclassifier")
        for key, value in self.classifier_dict.items():
            print(("key: " + key).ljust(70, ' '), end="")
            if (type(value) == float):
                print(value)
            else:
                print(value.shape)
        print()


if __name__ == '__main__':
    print("start")
    model_deployment = Model_deployment()
    model_deployment.load_tensor()
    # model_deployment.print_tensor()
    model_deployment.deploy_model()
    print('All const tensor size: {:.2f} KB'.format(model_deployment.file_writer.const_tensor_size / 1024))