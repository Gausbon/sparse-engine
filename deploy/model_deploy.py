import os
import numpy as np
import yaml
from layer_deploy import Layer_deployer

class Model_deployer(Layer_deployer):
    def __init__(self):
        super().__init__()

        # dict list
        self.dmt_count = self.config['dmt_count']
        assert(self.dmt_count == 1 or self.dmt_count == 2)
        if (self.dmt_count == 1):
            self.downsample_list_dict = {}
            self.mv2block_list_dict = {}
            self.transformer_list_dict = {}
        else:
            self.downsample_list_dict_1 = {}
            self.downsample_list_dict_2 = {}
            self.mv2block_list_dict_1 = {}
            self.mv2block_list_dict_2 = {}
            self.transformer_list_dict_1 = {}
            self.transformer_list_dict_2 = {}

        self.last_conv_dict = {}
        self.pooling_list_dict = {}
        self.classifier_dict = {}

        self.image = np.load(self.config['image_path']).transpose(0, 2, 3, 1)
        self.image_class = self.config['image_class']
        self.size = self.image.shape[1]
        
        print('-'*70)
        print('basic info')
        print('model config path: ' + str(self.config['model_config_path']))
        print('ram size: ' + str(self.max_size))
        print('image input path: ' + str(self.config['image_path']))
        print('image class: ' + str(self.image_class))
        print('force sparse: ' + str(self.config['force_sparse']))
        print('-'*70)


    def load_tensor(self):
        tensor_dir = os.listdir(self.tensor_path)
        
        for file in tensor_dir:
            name_list = file.split('.')
            if (name_list[0].startswith("downsample")):
                if (self.dmt_count == 1):
                    list_dict = self.downsample_list_dict
                elif (name_list[0].endswith('one')):
                    list_dict = self.downsample_list_dict_1
                else:
                    list_dict = self.downsample_list_dict_2
                
                if (name_list[1] in list_dict.keys()):
                    block_dict = list_dict[name_list[1]]
                else:
                    block_dict = {}
                    list_dict[name_list[1]] = block_dict

            elif (name_list[0].startswith("mv2block")):
                if (self.dmt_count == 1):
                    list_dict = self.mv2block_list_dict
                elif (name_list[0].endswith('one')):
                    list_dict = self.mv2block_list_dict_1
                else:
                    list_dict = self.mv2block_list_dict_2

                if (name_list[1] in list_dict.keys()):
                    block_dict = list_dict[name_list[1]]
                else:
                    block_dict = {}
                    list_dict[name_list[1]] = block_dict

            elif (name_list[0].startswith("transformer")):
                if (self.dmt_count == 1):
                    list_dict = self.transformer_list_dict
                elif (name_list[0].endswith('one')):
                    list_dict = self.transformer_list_dict_1
                else:
                    list_dict = self.transformer_list_dict_2

                if (name_list[1] in list_dict.keys()):
                    block_dict = list_dict[name_list[1]]
                else:
                    block_dict = {}
                    list_dict[name_list[1]] = block_dict

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
            if (name_list[0].startswith("downsample") or name_list[0].startswith("mv2block")
                or name_list[0].startswith("transformer")):
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
        if (self.dmt_count == 1):
            self.downsample_list_dict = dict(sorted(self.downsample_list_dict.items(), 
                    key=lambda x: int(x[0].split('_')[-1])))
            self.mv2block_list_dict = dict(sorted(self.mv2block_list_dict.items(), 
                    key=lambda x: int(x[0].split('_')[-1])))
            self.transformer_list_dict = dict(sorted(self.transformer_list_dict.items(), 
                    key=lambda x: int(x[0].split('_')[-1])))
        else:
            self.downsample_list_dict_1 = dict(sorted(self.downsample_list_dict_1.items(), 
                    key=lambda x: int(x[0].split('_')[-1])))
            self.downsample_list_dict_2 = dict(sorted(self.downsample_list_dict_2.items(), 
                    key=lambda x: int(x[0].split('_')[-1])))
            self.mv2block_list_dict_1 = dict(sorted(self.mv2block_list_dict_1.items(), 
                    key=lambda x: int(x[0].split('_')[-1])))
            self.mv2block_list_dict_2 = dict(sorted(self.mv2block_list_dict_2.items(), 
                    key=lambda x: int(x[0].split('_')[-1])))
            self.transformer_list_dict_1 = dict(sorted(self.transformer_list_dict_1.items(), 
                    key=lambda x: int(x[0].split('_')[-1])))
            self.transformer_list_dict_2 = dict(sorted(self.transformer_list_dict_2.items(), 
                    key=lambda x: int(x[0].split('_')[-1])))


    def deploy_self_attn(self, name:str, block_dict:dict,
            b:int, n:int, c:int, head_nums:int, section:str):
        
        # self_attention start
        # qkv linear: head -> tail
        # from (b n c) to (b n 3c)
        bncx3_size = self.size_list.pop(0)[1]
        self.input_dict['n'] = b * n
        self.deploy_linear(name + 'qqkv.', block_dict, True,
                    section, '&'+section+'['+str(self.max_size-bncx3_size)+']')
        
        # bnc transpose: tail -> head
        # from (b, n, 3, head_nums, c / head_nums) 
        # to (3, head_nums, b, n, c / head_nums)
        # n = input_size * input_size (image size), c = embedding_dim (channel)
        # current memory: [..... q k v | reserve]
        self.size_list.pop(0)
        self.deploy_transpose((b * n), (3 * head_nums),
                    (c / head_nums), '&'+section+'['+str(self.max_size-bncx3_size)+']',
                    section)
        self.file_writer.writeln('memcpy(&' + section + '['  + str(self.max_size-bncx3_size)
                    + '],' + section + ',' + str(bncx3_size) + ');', 'func')

        # q * k^T
        # from (head_nums*b, n, c / head_nums) * (head_nums*b, c / head_nums, n)
        # to (head_nums*b, n, n)
        # current memory: [q*k ... q k v | reserve]
        attn_size = self.size_list.pop(0)[1]
        self.deploy_matmul('self_attn.qmatmul_qk.', block_dict, (head_nums * b),
                    n, (c / head_nums),
                    n, True,
                    '&' + section + '[' + str(self.max_size - bncx3_size)+ ']',  # q
                    '&' + section + '[' + str(self.max_size - bncx3_size*2//3)+ ']',  # k
                    section)

        # in_place softmax
        self.size_list.pop(0)
        self.deploy_softmax('self_attn.qsoftmax1.', block_dict, 
                    head_nums * b * n, 
                    n, 
                    section, section)

        # attn * v
        # from (head_nums*b, n, n) * (head_nums*b, n, c / head_nums)
        # to (head_nums*b, n, c / head_nums)
        # current memory: [attn attn*v ... v | reserve]
        self.size_list.pop(0)
        self.deploy_matmul('self_attn.qmatmul_attnv.', block_dict, (head_nums * b),
                    n, n, (c / head_nums), False,
                    section, # attn
                    '&' + section + '[' + str(self.max_size - bncx3_size//3)+ ']',  # v
                    '&' + section + '[' + str(attn_size)+ ']')

        # bnc transpose
        # from (head_nums*b, n, c / head_nums)
        # to (b, n, c)
        # current memory: [(attn) attn*v ... bnc_value | reserve]
        bnc_size = self.size_list.pop(0)[1]
        self.deploy_transpose(head_nums, (b * n),
                    (c / head_nums), '&' + section + '[' + str(attn_size)+ ']',
                    '&' + section + '[' + str(self.max_size - bnc_size) + ']')

        # self attention final linear
        # current memory: [proj ... bnc_value | reserve]
        self.size_list.pop(0)
        self.input_dict['n'] = b * n
        self.deploy_linear('self_attn.qproj.', block_dict, True,
                    '&' + section + '[' + str(self.max_size - bnc_size) + ']', section)


    def deploy_tokenizer(self, batch:int, block_dict:dict,
            input_size:int, section:str):
        # qconv_relu: head -> tail
        size_0 = self.size_list.pop(0)[1]
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 1, 1
        self.deploy_conv(self, 'qconv_relu.', block_dict, False, False,
                    section, '&'+section+'['+str(self.max_size - size_0[1])+']')

        # max_pool: tail -> head
        self.size_list.pop(0)
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


    def deploy_mv2block(self, batch:int, name:str, block_dict:dict, is_sparse:bool,
            input_size:int, section:str):

        # config_list: expansion, stride, is_sparse, output_size
        # name: {*}mv2block{type}_{index} in mv2block
        # {*}mv2block_{index} in downsample
        type_name = name.split('_')[-2]
        if (type_name.startswith('downsample')):
            config_list = [2, 2, is_sparse, input_size/2]
        elif (type_name.endswith('0')):
            config_list = [2, 1, True, input_size]
        elif (type_name.endswith('1')):
            config_list = [3, 1, True, input_size]
        elif (type_name.endswith('2')):
            config_list = [4, 1, True, input_size]
        else:
            print('Cannot parse block as mv2block: ' + name)
            return

        res = False
        if (config_list[1] == 1):
            inp = int(block_dict['qconv.0.conv_module.weight'].shape[1])
            oup = int(block_dict['qconv.2.conv_module.weight'].shape[0])
            if (inp == oup and config_list[-1] == input_size):
                res = True
        
        if (res):
            feature_size = int(inp * input_size * input_size)
            self.file_writer.writeln('memcpy(&' + section + '[' + str(self.max_size - feature_size) 
                + '],' + section + ',' + str(feature_size) + ');', 'func')
            self.max_size -= feature_size

        # conv_0: head -> tail
        conv0_size = self.size_list.pop(0)[1]
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 0, 0

        self.deploy_conv('qconv.0.', block_dict, config_list[2], False,
                    section, '&'+section+'['+str(self.max_size - conv0_size)+']')

        # conv_1: tail -> head
        self.size_list.pop(0)
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.output_dict['h'], self.output_dict['w'] = config_list[3], config_list[3]
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = config_list[1], config_list[1]
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 1, 1

        self.deploy_conv('qconv.1.', block_dict, config_list[2], True,
                '&'+section+'['+str(self.max_size - conv0_size)+']', section)

        # conv_2: head -> tail
        conv2_size = self.size_list.pop(0)[1]
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = config_list[3], config_list[3]
        self.output_dict['h'], self.output_dict['w'] = config_list[3], config_list[3]
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 0, 0

        self.deploy_conv('qconv.2.', block_dict, config_list[2], False,
                    section, '&'+section+'['+str(self.max_size - conv2_size)+']')

        if (res):
            self.size_list.pop(0)
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
        conv1_size = self.size_list.pop(0)[1]
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 1, 1

        self.deploy_conv('qconv1.', block_dict, True, False,
                    section, '&'+section+'['+str(self.max_size - conv1_size)+']')

        # conv_2: tail -> head
        bnc_size = self.size_list.pop(0)[1]
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
        self.size_list.pop(0)
        self.deploy_norm('qpre_norm.', block_dict, norm_batch, embedding_dim,
                    section, section)

        # self attention
        self.deploy_self_attn('self_attn.', block_dict,
            batch, (input_size * input_size), embedding_dim, head_nums, section)
        
        # add1: head + tail -> mid
        self.max_size += bnc_size
        self.size_list.pop(0)
        self.deploy_add('qadd1.', block_dict, section, 
            '&'+section+'['+str(self.max_size-bnc_size)+']',
            '&'+section+'['+str(bnc_size)+']',bnc_size)

        # copy to head
        self.file_writer.writeln('memcpy(' + section + ',&'
                    + section+'['+str(bnc_size)+'],' + str(bnc_size) + ');', 'func')

        # norm1: head -> head
        self.size_list.pop(0)
        self.deploy_norm('qnorm1.', block_dict, norm_batch, 
                    embedding_dim, section, section)

        # copy head to shortcut
        self.file_writer.writeln('memcpy(&' + section + '[' + str(self.max_size-bnc_size) + '],'
                + section + ',' + str(bnc_size) + ');', 'func')
        self.max_size -= bnc_size
        
        # linear relu1: head -> tail
        linear_size = self.size_list.pop(0)[1]
        self.input_dict['n'] = norm_batch
        self.deploy_linear('qlinear_relu1.', block_dict, True,
                    section, '&'+section+'['+str(self.max_size-linear_size)+']')

        # linear2: tail -> head
        linear2_size = self.size_list.pop(0)[1]
        self.input_dict['n'] = norm_batch
        self.deploy_linear('qlinear2.', block_dict, True,
                    '&'+section+'['+str(self.max_size-linear_size)+']', section)

        # add2: head + tail -> mid
        self.max_size += bnc_size
        self.size_list.pop(0)
        self.deploy_add('qadd2.', block_dict,
            section,
            '&'+section+'['+str(self.max_size-bnc_size)+']', 
            '&'+section+'['+str(linear2_size)+']',linear2_size)

        # copy add2 to head
        self.file_writer.writeln('memcpy(' + section + ',&' + section +
                '[' + str(linear2_size) + '],' + str(linear2_size) + ');', 'func')

        # conv_3: head -> tail
        conv3_size = self.size_list.pop(0)[1]
        self.input_dict['n'] = batch
        self.input_dict['h'], self.input_dict['w'] = input_size, input_size
        self.output_dict['h'], self.output_dict['w'] = input_size, input_size
        self.conv_dict['stride.h'], self.conv_dict['stride.w'] = 1, 1
        self.conv_dict['padding.h'], self.conv_dict['padding.w'] = 1, 1

        self.deploy_conv('qconv3.', block_dict, True, False,
                    section, '&'+section+'['+str(self.max_size-conv3_size)+']')

        # conv_4: tail -> head
        self.size_list.pop(0)
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
        conv_size = self.size_list.pop(0)[1]
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
            all_size = self.size_list.pop(0)
            in_size, out_size = all_size[0], all_size[1]
            self.file_writer.writeln('memcpy(&' + section + '[' + str(self.max_size - in_size)
                + '],' + section + ',' + str(in_size) + ');', 'func')
            self.max_size -= in_size

            # linear: head -> tail
            # shape: b (h w) c -> b (h w) 1 -> b (h w) 1
            self.input_dict['n'] = batch * input_size * input_size
            self.deploy_linear('qattention_pool.', block_dict, False,
                        section, '&'+section+'['+str(self.max_size - out_size)+']')
            
            # qsoftmax: tail -> tail
            self.size_list.pop(0)
            self.deploy_softmax('qsoftmax.', block_dict, batch, (input_size * input_size), 
                        '&'+section+'['+str(self.max_size - out_size)+']', 
                        '&'+section+'['+str(self.max_size - out_size)+']')

            # matmul: tail * tail -> mid
            # shape: b 1 (h w) * b (h w) c -> b 1 c -> b c
            self.size_list.pop(0)
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
        linear_size = self.size_list.pop(0)[1]
        self.input_dict['n'] = batch
        self.deploy_linear('', block_dict, False,
                        section, '&'+section+'['+str(self.max_size-linear_size)+']')

        # copy from tail to head
        self.file_writer.writeln('memcpy(' + section + ',&' + section 
            +'['+str(self.max_size-linear_size)+'],' + str(linear_size) + ');', 'func')

        print('Block:' + name + ' deploy completed')


    def dedploy_dmt(self, batch, downsample_list_dict, mv2block_list_dict, 
            transformer_list_dict, d_sparse):
        for key, value in downsample_list_dict.items():
            self.file_writer.writeln('// block: ' + key, 'func')
            key_list = key.split('_')
            if (key_list[0].endswith('0')):
                self.deploy_tokenizer(batch, value, self.size, 'section')
            elif (key_list[0].endswith('1')):
                self.deploy_mv2block(batch, key, value, d_sparse, self.size, 'section')
            self.file_writer.writeln('printf("'+ key + ' finished\\r\\n");', 'func')
            self.size /= 2
            print('-'*70)

        for key, value in mv2block_list_dict.items():
            self.file_writer.writeln('// block: ' + key, 'func')
            self.deploy_mv2block(batch, key, value, d_sparse, self.size, 'section')
            self.file_writer.writeln('printf("'+ key + ' finished\\r\\n");', 'func')
            print('-'*70)

        for key, value in transformer_list_dict.items():
            self.file_writer.writeln('// block: ' + key, 'func')
            self.deploy_transformer(batch, key, value, self.size, 'section')
            self.file_writer.writeln('printf("'+ key + ' finished\\r\\n");', 'func')
            print('-'*70)


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
        
        self.file_writer.write_init(self.image, self.block, self.max_size)

        # deploy quantization inference
        if (self.config['dmt_count'] == 1):
            self.dedploy_dmt(batch, self.downsample_list_dict, self.mv2block_list_dict, 
                self.transformer_list_dict)
        else:
            self.dedploy_dmt(batch, self.downsample_list_dict_1, self.mv2block_list_dict_1, 
                self.transformer_list_dict_1, False)
            self.dedploy_dmt(batch, self.downsample_list_dict_2, self.mv2block_list_dict_2, 
                self.transformer_list_dict_2, True)

        self.file_writer.writeln('// block: last conv', 'func')
        self.deploy_last_conv(batch, 'last_conv', self.last_conv_dict,
            self.size, section)
        self.file_writer.writeln('printf("last_conv finished\\r\\n");', 'func')
        print('-'*70)

        for key, value in self.pooling_list_dict.items():
            self.file_writer.writeln('// block: ' + key, 'func')
            self.deploy_global_pooling(batch, key, value,
                self.size, last_channel, section)
            self.file_writer.writeln('printf("'+ key + ' finished\\r\\n");', 'func')
            print('-'*70)

        self.file_writer.writeln('// block: classifier', 'func')
        self.deploy_classifier(batch, 'classifier', self.classifier_dict,
            section)
        self.file_writer.writeln('printf("classifier finished\\r\\n");', 'func')
        print('-'*70)
        
        self.file_writer.write_end(self.image_class, self.max_buf_size, self.max_quant_size)

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
    print('-'*70)
    print("start")
    model_deployer = Model_deployer()
    model_deployer.load_tensor()
    # model_deployer.print_tensor()
    model_deployer.deploy_model()
    print('All const tensor size: {:.2f} KB'.format(model_deployer.file_writer.const_tensor_size / 1024))