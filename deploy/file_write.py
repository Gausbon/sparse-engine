class File_writer():
    def __init__(self, func_path, data_path):
        self.func_path = func_path
        self.data_path = data_path
        self.func_file = open(self.func_path,'w')
        self.data_file = open(self.data_path,'w')
        self.var_dict = {}


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

        w_file.write('	  ' + str + '\n\n')


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


    def write_tensor(self, tensor, name, is_static, data_type):
        if (is_static):
            self.data_file.write("static ")
        self.data_file.write(data_type + ' ' + name + '[' + str(tensor.size+1) + '] = {')
        tensor_flatten = tensor.flatten()
        for i in range(0, tensor_flatten.size):
            # tensor must be int
            self.data_file.write(str(int(tensor_flatten[i])))
            if (i != tensor_flatten.size - 1):
                self.data_file.write(',')
        self.data_file.write('};\n\n')


    def write_func_call(self, name, param_list):
        self.func_file.write('	  ' + name + '(')
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
                self.func_file.write('	  ' + name + '.' + key + '=' + str(value) + ';\n')
        self.func_file.write('\n')