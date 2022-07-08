import numpy as np
# for the sparse encode algorithm, please check the readme.md


def approximate_float(M):
    significand, shift = np.frexp(M)
    significand_q31 = np.round(significand * (1 << 31))
    return significand_q31, shift


def conv_data_to_sparse(input):
    return conv_data_to_sparse_encode_1(input)

    
# encode using algorithm 1
def conv_data_to_sparse_encode_1(input):
    filter_list = []
    if (len(input.shape) == 4):
        info_list = [0, 0, 0, 0]
        third_dim_count = input.shape[3]
        second_dim_count = third_dim_count * input.shape[2]
        first_dim_count = second_dim_count * input.shape[1]
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                for k in range(input.shape[2]):
                    for l in range(input.shape[3]):
                        if (input[i][j][k][l] != 0):
                            col_dis = (i - info_list[0]) * first_dim_count \
                                        + (j - info_list[1]) * second_dim_count \
                                        + (k - info_list[2]) * third_dim_count \
                                        + (l - info_list[3])    
                            while (col_dis >= 256):
                                col_dis -= 255
                                filter_list.extend([127, 0])
                            filter_list.extend([col_dis-128, input[i][j][k][l]])
                            info_list = [i, j, k, l]
    elif (len(input.shape) == 4):
        info_list = [0, 0]
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                if (input[i][j] != 0):
                    col_dis = + (i - info_list[0]) * input.shape[1] \
                                + (j - info_list[1])
                    while (col_dis >= 256):
                        col_dis -= 255
                        filter_list.extend([127, 0])
                    filter_list.extend([col_dis-128, input[i][j]])
                    info_list = [i, j]
    else:
        print('wrong filter shape! ' + str(input.shape))
    return filter_list


# decode using algorithm 1
def conv_data_to_sparse_decode_1(encode_list, zero_list):
    for i in range(0, len(encode_list)):
        encode_matrix = encode_list[i]
        zero_matrix = zero_list[i]
        shape = zero_matrix.shape
        pos = [0, 0, 0, 0]

        for j in range(0, len(encode_matrix) // 2):
            pos[3] += (encode_matrix[2*j] + 128)
            while (pos[3] >= shape[3]):
                pos[3] -= shape[3]
                pos[2] += 1
                while (pos[2] >= shape[2]):
                    pos[2] -= shape[2]
                    pos[1] += 1
                    while (pos[1] >= shape[1]):
                        pos[1] -= shape[1]
                        pos[0] += 1
                        if (pos[0] >= shape[0]):
                            raise Exception("out of bound error, pos:" + str(pos))
            zero_matrix[pos[0]][pos[1]][pos[2]][pos[3]] = encode_matrix[2*j+1]


# todo: encode using algorithm 2
def conv_data_to_sparse_encode_2(list):
    raise Exception("Currently does not support algorithm 2 encode")


# todo: decode using algorithm 2
def conv_data_to_sparse_decode_2(encode_list, zero_list):
    raise Exception("Currently does not support algorithm 2 decode")