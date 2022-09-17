import numpy as np
import yaml
from utils import conv_data_to_sparse, approximate_float, get_addr_str
from file_write import File_writer

class Layer_deployer():
    def __init__(self):

        # basic info
        with open('config.yml', 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        self.tensor_path = '../TinySPOS/' + self.config['tensor_path']
        self.file_writer = File_writer(self.config['func_path'], self.config['data_path'])
        with open('../TinySPOS/' + self.config['memory_list'], 'r') as file:
            self.size_list = yaml.load(file, Loader=yaml.FullLoader)
        
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

        self.force_sparse = self.config['force_sparse']
        self.block = self.config['block']
        
        # dynamic size
        self.max_quant_size = 0
        self.max_buf_size = 0
        self.max_size = 0

        for layer in self.size_list:
            sum = 0
            for item in layer:
                sum += item
            if sum > self.max_size:
                self.max_size = sum
        self.ori_max_size = self.max_size
                

    def deploy_add(self, name:str, block_dict:dict, 
            in_addr_1:int, in_addr_2:int, out_addr:int, count:int):
        
        # init
        param_list = [get_addr_str(in_addr_1), get_addr_str(in_addr_2)]

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
            0, get_addr_str(out_addr), qo_offset, 0x7fffffff, 0, -128, 127, count])
        self.file_writer.write_func_call('arm_elementwise_add_s8_with_neg', param_list)

        self.file_writer.write_extime('add')


    def deploy_conv(self, name:str, block_dict:dict, is_sparse:bool, is_depthwise:bool, 
            in_addr:int, out_addr:int):

        # init
        param_list = ['&ctx']
        if (is_depthwise):
            param_list.append('&dw_conv_params')
        else:
            param_list.append('&conv_params')
        param_list.extend(['&c_quant_params', '&input_dims', get_addr_str(in_addr)])

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
        if (weight.shape[0] > self.max_quant_size):
            self.max_quant_size = weight.shape[0]
        
        self.output_dict['c'] = ori_shape[0]
        if (is_depthwise and not is_sparse):
            self.input_dict['c'] = ori_shape[0]
        else:
            self.input_dict['c'] = ori_shape[1]
        self.filter_dict['h'], self.filter_dict['w'] = ori_shape[2], ori_shape[3]
        sparse_buf_size = 4 * self.output_dict['h'] * self.output_dict['w']
        none_sparse_buf_size = 4 * self.input_dict['c'] * self.filter_dict['h'] * self.filter_dict['w']

        # sparse encode
        if (is_depthwise and not is_sparse):
            weight = weight.transpose((1, 2, 3, 0))
        else:
            weight = weight.transpose((0, 2, 3, 1))

        # recheck the sparse encode 
        if (is_sparse):
            if (not is_depthwise):
                weight, is_sparse = conv_data_to_sparse(weight, self.block, self.force_sparse)
            else:
                # dw sparse conv use block 3 at w direction
                weight, is_sparse = conv_data_to_sparse(weight, 3, self.force_sparse)
            
            # this branch goes to dwconv:
            # should be sparse, but after sparse encode the tensor got larger
            if (is_depthwise and not is_sparse):
                weight = weight.transpose((1, 2, 3, 0))
                self.input_dict['c'] = ori_shape[0]

        # record the buffer size
        if is_sparse:
            buf_size = sparse_buf_size
        else:
            buf_size = none_sparse_buf_size
        
        if (buf_size > self.max_buf_size):
            self.max_buf_size = buf_size

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
        param_list.extend(['&output_dims', get_addr_str(out_addr)])
        self.conv_dict['activation.max'] = 127
        self.conv_dict['activation.min'] = -128
        # notice the input offset is negative zero point
        self.conv_dict['input_offset'] = -block_dict[name + 'qi.zero_point']
        self.conv_dict['output_offset'] = block_dict[name + 'qo.zero_point']
        
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
        self.file_writer.write_const_tensor(shift, shift_name, 'q7_t')
        
        self.file_writer.writeln('memcpy(conv_mult_use,' + mult_name + ',' + str(4*mult.size) + ');','func')
        self.file_writer.writeln('for(int i = 0; i < ' + str(shift.size) + '; i++) { conv_shift_use[i] = ' + shift_name + '[i]; }', 'func')

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
        self.file_writer.write_extime('conv')


    def deploy_linear(self, name:str, block_dict:dict, is_sparse:bool,
            in_addr:int, out_addr:int):

        # init
        param_list = ['&ctx', '&fc_params', '&t_quant_params', '&input_dims', 
            get_addr_str(in_addr)]
        
        # weight operation
        weight = block_dict[name + 'fc_module.weight']
        weight_shape = weight.shape
        if (is_sparse):
            weight, is_sparse = conv_data_to_sparse(weight, self.block, self.force_sparse)
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
        param_list.extend(['&output_dims', get_addr_str(out_addr)])
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
        self.file_writer.write_extime('linear')


    def deploy_pooling(self, name:str, block_dict:dict, is_avg:bool,
            in_addr:int, out_addr:int):

        # init
        param_list = ['&ctx', '&pool_params', '&input_dims', get_addr_str(in_addr), 
                '&filter_dims', '&output_dims']
        
        self.file_writer.write_param_parser('input_dims', self.input_dict)
        self.file_writer.write_param_parser('filter_dims', self.filter_dict)
        self.file_writer.write_param_parser('output_dims', self.output_dict)
        self.file_writer.write_param_parser('pool_params', self.pooling_dict)

        if (4 * self.input_dict['c'] > self.max_buf_size):
            self.max_buf_size = 4 * self.input_dict['c']

        # get offset
        param_list.append(-block_dict[name + 'qi.zero_point'])
        if (is_avg):
            param_list.append(block_dict[name + 'qo.zero_point'])
        else:
            # qi.zero_point = qo.zero_point
            param_list.append(block_dict[name + 'qi.zero_point'])
        
        # get input mult
        qi_scale = block_dict[name + 'qi.scale']
        if (is_avg):
            qo_scale = block_dict[name + 'qo.scale']
        else:
            qo_scale = block_dict[name + 'qi.scale']
            
        mult, shift = approximate_float(qi_scale / qo_scale)
        param_list.extend([mult, shift, get_addr_str(out_addr)])

        # func call
        func_name = 'arm_'
        if (is_avg):
            func_name = func_name + 'avgpool_s8_with_quantization'
        else:
            func_name = func_name + 'maxpool_s8_with_quantization'

        self.file_writer.write_func_call(func_name, param_list)
        self.file_writer.write_extime('pool')


    def deploy_matmul(self, name:str, block_dict, dim_b:int, dim_lr:int, dim_lc:int,
            dim_rc:int, is_trans:bool, in_addr_1:int, in_addr_2:int, out_addr:int):
        
        # init
        param_list = [get_addr_str(in_addr_1), get_addr_str(in_addr_2), 'NULL', get_addr_str(out_addr)]
        if (not is_trans):
            param_list.insert(0, '&ctx')
            if (4 * dim_rc > self.max_buf_size):
                self.max_buf_size = 4 * dim_rc

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

        self.file_writer.write_extime('matmul')


    def deploy_norm(self, name:str, block_dict:dict, dim_b:int, dim_c:int, 
            in_addr:int, out_addr:int):

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

        param_list.extend([get_addr_str(in_addr), get_addr_str(out_addr)])
        self.file_writer.write_func_call('arm_layernorm_s8', param_list)

        self.counter += 1
        self.file_writer.write_extime('norm')


    def deploy_transpose(self, dim_b:int, dim_h:int, dim_w:int, dim_c:int,
            in_addr:int, out_addr:int):

        param_list = [dim_b, dim_h, dim_w, dim_c, get_addr_str(in_addr), get_addr_str(out_addr)]
        self.file_writer.write_func_call('arm_nn_transpose_bhwc_to_bwhc_q7', param_list)
        self.file_writer.write_extime('trans')


    def deploy_softmax(self, name:str, block_dict, dim_b:int, dim_c:int,
            in_addr:int, out_addr:int):

        # init
        param_list = ['&ctx', get_addr_str(in_addr), dim_b, dim_c]

        # input quant
        qi_scale = block_dict[name + 'qi.scale']
        softmax_input_integer_bits = 5
        qi_scale = min(qi_scale * (1 << (31 - softmax_input_integer_bits)),
                                   (1 << 31) - 1)
        qi_mult, qi_shift = approximate_float(qi_scale)
        
        if (256 * 4 + 256 > self.max_buf_size):
            self.max_buf_size = 256 * 4 + 256

        param_list.extend([qi_mult, qi_shift, -248, 
                get_addr_str(out_addr)])
        self.file_writer.write_func_call('arm_softmax_s8_fast', param_list)

        self.file_writer.write_extime('softmax')

