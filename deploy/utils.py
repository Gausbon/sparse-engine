import numpy as np
# for the sparse encode algorithm, please check the readme.md


def get_addr_str (addr:int):
    if (addr == 0):
        return 'section'
    else:
        return '&section[' + str(addr) + ']'

'''
def get_sparsity(M):
    M_flatten = M.flatten()
    zero_count = 0
    for item in M_flatten:
        if (item == 0):
            zero_count += 1
    print(zero_count / M_flatten.size)
'''

def approximate_float(M):
    significand, shift = np.frexp(M)
    significand_q31 = np.round(significand * (1 << 31))
    if (type(significand_q31) == np.float64):
        significand_q31 = int(significand_q31)
    if (type(shift) == np.float64):
        shift = int(shift)
    return significand_q31, shift


def conv_data_to_sparse(input, block, sparse_choice):
    # get_sparsity(input)
    if (sparse_choice == 0):
        return input, False
    output = conv_data_to_sparse_encode_1(input, block)
    # print('in: ' + str(input.size))
    # print('out: ' + str(output.size))
    if (sparse_choice == 1 or output.size < input.size):
        return output, True
    else:
        return input, False
    

# encode using algorithm 1
def conv_data_to_sparse_encode_1(input, block):
    filter_list = []
    input_f = input.flatten()
    last_pos = 0
    block_cnt = 0
    for i in range(0, len(input_f)):
        if (block_cnt != 0):
            filter_list.append(input_f[i])
            block_cnt = (block_cnt + 1) % block
            if (block_cnt == 0):
                last_pos = i
        else:
            if (input_f[i] == 0):
                continue
            else:
                col_dis = i - last_pos
                while (col_dis >= 256):
                    col_dis -= 255
                    filter_list.extend([127, 0])
                filter_list.extend([col_dis-128, input_f[i]])
                block_cnt = (block_cnt + 1) % block
    '''
    if (len(input.shape) == 4):
        info_list = [0, 0, 0, 0]
        third_dim_count = input.shape[3]
        second_dim_count = third_dim_count * input.shape[2]
        first_dim_count = second_dim_count * input.shape[1]
        i = 0
        while i < input.shape[0]:
            j = 0
            while j < input.shape[1]:
                k = 0
                while k < input.shape[2]:
                    l = 0
                    while l < input.shape[3]:
                        if (input[i][j][k][l] != 0):
                            col_dis = (i - info_list[0]) * first_dim_count \
                                        + (j - info_list[1]) * second_dim_count \
                                        + (k - info_list[2]) * third_dim_count \
                                        + (l - info_list[3])    
                            while (col_dis >= 256):
                                col_dis -= 255
                                filter_list.extend([127, 0])
                            filter_list.extend([col_dis-128, input[i][j][k][l]])
                            for _ in range(0, block-1):
                                l += 1
                                if (l >= input.shape[3]):
                                    k += (l // input.shape[3])
                                    l = l % input.shape[3]
                                    if (k >= input.shape[2]):
                                        j += (k // input.shape[2])
                                        k = k % input.shape[2]
                                        if (j >= input.shape[1]):
                                            i += (j // input.shape[1])
                                            j = j % input.shape[1]
                                            if (i >= input.shape[0]):
                                                return np.array(filter_list)
                                filter_list.append(input[i][j][k][l])

                            info_list = [i, j, k, l]
                        l += 1
                    k += 1
                j += 1
            i += 1
        
    elif (len(input.shape) == 2):
        info_list = [0, 0]
        i = 0
        while i < input.shape[0]:
            j = 0
            while j < input.shape[1]:
                if (input[i][j] != 0):
                    col_dis = + (i - info_list[0]) * input.shape[1] \
                                + (j - info_list[1])
                    while (col_dis >= 256):
                        col_dis -= 255
                        filter_list.extend([127, 0])
                    filter_list.extend([col_dis-128, input[i][j]])
                    for _ in range(0, block-1):
                        j += 1
                        if (j >= input.shape[1]):
                            i += (j // input.shape[1])
                            j = j % input.shape[1]
                            if (i >= input.shape[0]):
                                return np.array(filter_list)
                        filter_list.append(input[i][j])

                    info_list = [i, j]
                j += 1
            i += 1

    else:
        print('wrong filter shape! ' + str(input.shape))
    '''
    return np.array(filter_list)


# todo: encode using algorithm 2
def conv_data_to_sparse_encode_2(list):
    raise Exception("Currently does not support algorithm 2 encode")
