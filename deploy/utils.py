import numpy as np
# for the sparse encode algorithm, please check the readme.md


def approximate_float(M):
    significand, shift = np.frexp(M)
    significand_q31 = np.round(significand * (1 << 31))
    if (type(significand_q31) == np.float64):
        significand_q31 = int(significand_q31)
    if (type(shift) == np.float64):
        shift = int(shift)
    return significand_q31, shift


def conv_data_to_sparse(input, block, force_sparse):
    output = conv_data_to_sparse_encode_1(input, block)
    print('out: ' + str(output.size))
    print('in: ' + str(input.size))
    if (force_sparse or output.size < input.size):
        return output, True
    else:
        return input, False
    

# encode using algorithm 1
def conv_data_to_sparse_encode_1(input, block):
    filter_list = []
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
    return np.array(filter_list)


# todo: encode using algorithm 2
def conv_data_to_sparse_encode_2(list):
    raise Exception("Currently does not support algorithm 2 encode")
