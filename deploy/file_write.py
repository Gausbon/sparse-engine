from numpy import ndarray


class File_writer():
    def __init__(self, func_path, data_path):
        self.func_path = func_path
        self.data_path = data_path
        self.func_file = open(self.func_path,'w')
        self.data_file = open(self.data_path,'w')
        self.var_dict = {}
        self.const_tensor_size = 0
        self.init_pos = 0

    def __del__(self):
        self.func_file.close()
        self.data_file.close()


    def writeln(self, str, file_str):
        if (file_str == 'func'):
            w_file = self.func_file
        elif (file_str == 'data'):
            w_file = self.data_file
        else:
            return

        w_file.write('    ' + str + '\n\n')


    def write_file(self, r_file, file_str):
        lines = r_file.readlines()
        if (file_str == 'func'):
            w_file = self.func_file
        elif (file_str == 'data'):
            w_file = self.data_file
        else:
            return

        for item in lines:
            w_file.write(item)


    def write_const_tensor(self, tensor, name, data_type):
        self.data_file.write("static const " + data_type + ' ' + name + '[' + str(tensor.size) + '] = {')
        tensor_flatten = tensor.flatten()
        for i in range(0, tensor_flatten.size):
            # tensor must be int
            self.data_file.write(str(int(tensor_flatten[i])))
            if (i != tensor_flatten.size - 1):
                self.data_file.write(',')
        self.data_file.write('};\n\n')
        if (data_type == 'q31_t' or data_type == 'int32_t'):
            self.const_tensor_size += (4 * tensor.size)
        else:
            self.const_tensor_size += tensor.size

    def write_func_call(self, name, param_list):
        self.func_file.write('    ' + name + '(')
        for i in range(0, len(param_list)):
            if (type(param_list[i]) != float):
                self.func_file.write(str(param_list[i]))
            else:
                self.func_file.write(str(int(param_list[i])))
            if (i != len(param_list)-1):
                self.func_file.write(',')
        self.func_file.write(');\n\n')


    def write_param_parser(self, name, param_dict):
        for key, value in param_dict.items():
            if (type(value) == float):
                value = int(value)

            new_key = name + '.' + key
            if (new_key not in self.var_dict or value != self.var_dict[new_key]):
                self.var_dict[new_key] = value
                self.func_file.write('    ' + name + '.' + key + '=' + str(value) + ';\n')
        self.func_file.write('\n')

    
    def write_extime(self, type:str):
        self.func_file.write('    end = HAL_GetTick();\n')
        self.func_file.write('    ' + type + '_time += (end - start);\n')
        self.func_file.write('    ' + type + '_count++;\n')
        self.func_file.write('    start = end;\n\n')


    def write_init(self, image:ndarray, max_size:int):
        self.init_pos = self.func_file.tell()

        self.func_file.write('    static q7_t buf[0]={0};\n')
        self.func_file.write('    static int32_t conv_mult_use[0]={0};\n')
        self.func_file.write('    static int32_t conv_shift_use[0]={0};\n')
        self.func_file.write('        \n')

        self.func_file.write('    static q7_t section[' + str(max_size) + ']={0};\n\n')
        self.func_file.write('    c_quant_params.multiplier=conv_mult_use;\n')
        self.func_file.write('    c_quant_params.shift=conv_shift_use;\n\n')
        self.func_file.write('    ctx.size = sizeof(buf);\n')
        self.func_file.write('    ctx.buf = buf;\n\n')
        self.func_file.write('    memcpy(&section,&image,3072);\n\n')
        self.func_file.write('    start = HAL_GetTick();\n\n')
        
        self.write_const_tensor(image, 'image', 'q7_t')

    def write_end(self, image_class:int, buf_size:int, quant_size:int):
        result_check_params = ['&ctx','section','conv_count','conv_time',
            'linear_count','linear_time','trans_count','trans_time',
            'softmax_count','softmax_time','norm_count','norm_time',
            'pool_count','pool_time','matmul_count','matmul_time',
            'add_count','add_time',str(image_class)]
        self.write_func_call('result_check_statistics', result_check_params)

        self.func_file.write('    return 0;\n}\n')
        self.data_file.write('\n#endif\n')
        
        self.func_file.seek(self.init_pos)
        self.func_file.write('    static q7_t buf[' + str(int(buf_size)) + ']={0};\n')
        self.func_file.write('    static int32_t conv_mult_use[' + str(int(quant_size)) + ']={0};\n')
        self.func_file.write('    static int32_t conv_shift_use[' + str(int(quant_size)) + ']={0};\n')
        