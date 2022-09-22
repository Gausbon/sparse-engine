#include "arm_nnfunctions.h"
#include "data.h"
#include "func.h"
#include "stdio.h"
#include "main.h"

int quantization_inference(void) {
    uint32_t start, end;
    
    int32_t conv_count = 0, conv_time = 0;
    int32_t linear_count = 0, linear_time = 0;
    int32_t trans_count = 0, trans_time = 0;
    int32_t softmax_count = 0, softmax_time = 0;
    int32_t norm_count = 0, norm_time = 0;
    int32_t pool_count = 0, pool_time = 0;
    int32_t matmul_count = 0, matmul_time = 0;
    int32_t add_count = 0, add_time = 0;

    cmsis_nn_dims input_dims, output_dims, filter_dims, bias_dims;
    
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_conv_params  conv_params;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_layernorm_params norm_params;
    
    cmsis_nn_per_tensor_quant_params t_quant_params;
    cmsis_nn_per_channel_quant_params c_quant_params;

    cmsis_nn_context ctx;

    static q7_t buf[3456]={0};
    static int32_t conv_mult_use[512]={0};
    static int32_t conv_shift_use[512]={0};
 
    static q7_t section[229376]={0};

    c_quant_params.multiplier=conv_mult_use;
    c_quant_params.shift=conv_shift_use;

    ctx.size = sizeof(buf);
    ctx.buf = buf;

    memcpy(&section,&image,3072);

    start = HAL_GetTick();

    // block: downsample_mv2block_0

    input_dims.n=1;
    input_dims.h=32;
    input_dims.w=32;
    input_dims.c=3;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.h=32;
    output_dims.w=32;
    output_dims.c=6;

    conv_params.stride.h=1;
    conv_params.stride.w=1;
    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.activation.max=127;
    conv_params.activation.min=-128;
    conv_params.input_offset=5;
    conv_params.output_offset=-128;
    conv_params.dilation.h=1;
    conv_params.dilation.w=1;

    memcpy(conv_mult_use,mult_0,24);

    for(int i = 0; i < 6; i++) { conv_shift_use[i] = shift_0[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_0,&bias_dims,bias_0,&output_dims,&section[223232]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=6;

    filter_dims.h=3;
    filter_dims.w=3;

    output_dims.h=16;
    output_dims.w=16;

    dw_conv_params.stride.h=2;
    dw_conv_params.stride.w=2;
    dw_conv_params.padding.h=1;
    dw_conv_params.padding.w=1;
    dw_conv_params.activation.max=127;
    dw_conv_params.activation.min=-128;
    dw_conv_params.input_offset=128;
    dw_conv_params.output_offset=-128;
    dw_conv_params.dilation.h=1;
    dw_conv_params.dilation.w=1;
    dw_conv_params.ch_mult=1;

    memcpy(conv_mult_use,mult_1,24);

    for(int i = 0; i < 6; i++) { conv_shift_use[i] = shift_1[i]; }

    arm_depthwise_conv_s8(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[223232],&filter_dims,weight_1,&bias_dims,bias_1,&output_dims,section);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.h=16;
    input_dims.w=16;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=32;

    conv_params.input_offset=128;
    conv_params.output_offset=-7;

    memcpy(conv_mult_use,mult_2,128);

    for(int i = 0; i < 32; i++) { conv_shift_use[i] = shift_2[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_2,&bias_dims,bias_2,&output_dims,&section[221184]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(section,&section[221184],8192);

    printf("downsample_mv2block_0 finished\r\n");

    // block: mv2block0_0

    input_dims.c=32;


    output_dims.c=64;

    conv_params.input_offset=7;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_3,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_3[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_3,&bias_dims,bias_3,&output_dims,&section[212992]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=64;

    filter_dims.h=3;
    filter_dims.w=3;


    dw_conv_params.stride.h=1;
    dw_conv_params.stride.w=1;

    memcpy(conv_mult_use,mult_4,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_4[i]; }

    arm_depthwise_conv_s8(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[212992],&filter_dims,weight_4,&bias_dims,bias_4,&output_dims,section);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;


    conv_params.input_offset=128;
    conv_params.output_offset=3;

    memcpy(conv_mult_use,mult_5,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_5[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_5,&bias_dims,bias_5,&output_dims,&section[212992]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(section,&section[212992],16384);

    printf("mv2block0_0 finished\r\n");

    // block: mv2block0_1

    memcpy(&section[212992],section,16384);



    output_dims.c=128;

    conv_params.input_offset=-3;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_6,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_6[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_6,&bias_dims,bias_6,&output_dims,&section[180224]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=128;

    filter_dims.h=3;
    filter_dims.w=3;



    memcpy(conv_mult_use,mult_7,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_7[i]; }

    arm_depthwise_conv_s8(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[180224],&filter_dims,weight_7,&bias_dims,bias_7,&output_dims,section);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=64;

    conv_params.input_offset=128;
    conv_params.output_offset=3;

    memcpy(conv_mult_use,mult_8,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_8[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_8,&bias_dims,bias_8,&output_dims,&section[196608]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(&section[196608],&section[212992],-3,1410040064,0,-3,1708002304,0,0,section,1,2147483647,0,-128,127,16384);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    printf("mv2block0_1 finished\r\n");

    // block: transformer0_0

    input_dims.c=64;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=-1;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_9,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_9[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_9,&bias_dims,bias_9,&output_dims,&section[212992]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=96;

    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_10,384);

    for(int i = 0; i < 96; i++) { conv_shift_use[i] = shift_10[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,&section[212992],&filter_dims,weight_10,&bias_dims,bias_10,&output_dims,section);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(&section[204800],section,24576);

    norm_params.activation.max=127;
    norm_params.activation.min=-128;
    norm_params.input_offset=128;
    norm_params.output_offset=-81;

    t_quant_params.multiplier=1237536512;
    t_quant_params.shift=-8;

    arm_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,96,weight_11,bias_11,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    fc_params.activation.max=127;
    fc_params.activation.min=-128;
    fc_params.input_offset=81;
    fc_params.output_offset=-3;

    t_quant_params.multiplier=1294720384;

    input_dims.n=256;

    filter_dims.n=96;

    output_dims.c=288;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_12,&bias_dims,NULL,&output_dims,&section[131072]);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,6,48,&section[131072],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    memcpy(&section[131072],section,73728);

    arm_nn_batch_mat_mult_nt_t_s8(&section[131072],&section[155648],NULL,section,1103812096,-8,256,48,256,3,3,-2,2,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_softmax_s8_fast(&ctx,section,512,256,2044218880,22,-248,section);

    end = HAL_GetTick();
    softmax_time += (end - start);
    softmax_count++;
    start = end;

    arm_nn_batch_mat_mult_s8(&ctx,section,&section[180224],NULL,&section[131072],1118498432,-6,256,256,48,128,3,-1,2,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,2,256,48,&section[131072],&section[180224]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    fc_params.input_offset=1;
    fc_params.output_offset=-20;

    t_quant_params.multiplier=2040644608;
    t_quant_params.shift=-9;



    output_dims.c=96;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,&section[180224],&filter_dims,weight_13,&bias_dims,bias_13,&output_dims,section);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[204800],20,1975085824,-1,128,1859176576,0,0,&section[24576],-85,2147483647,0,-128,127,24576);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[24576],24576);

    norm_params.input_offset=85;
    norm_params.output_offset=-34;

    t_quant_params.multiplier=2111477760;

    arm_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,96,weight_14,bias_14,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    memcpy(&section[204800],section,24576);

    fc_params.input_offset=34;
    fc_params.output_offset=-128;

    t_quant_params.multiplier=1742290816;
    t_quant_params.shift=-7;



    output_dims.c=128;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_15,&bias_dims,bias_15,&output_dims,&section[172032]);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    fc_params.input_offset=128;
    fc_params.output_offset=0;

    t_quant_params.multiplier=1919606272;
    t_quant_params.shift=-9;


    filter_dims.n=128;

    output_dims.c=96;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,&section[172032],&filter_dims,weight_16,&bias_dims,bias_16,&output_dims,section);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[204800],0,1846320512,-1,34,1982880640,0,0,&section[24576],-29,2147483647,0,-128,127,24576);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[24576],24576);

    input_dims.n=1;
    input_dims.c=96;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=29;

    memcpy(conv_mult_use,mult_17,384);

    for(int i = 0; i < 96; i++) { conv_shift_use[i] = shift_17[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_17,&bias_dims,bias_17,&output_dims,&section[204800]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=128;

    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_18,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_18[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,&section[204800],&filter_dims,weight_18,&bias_dims,bias_18,&output_dims,section);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    printf("transformer0_0 finished\r\n");

    // block: transformer0_1

    input_dims.c=128;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;

    memcpy(conv_mult_use,mult_19,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_19[i]; }

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,128,1,section,&section[196608]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    arm_convolve_s8_sparse_1x1_CHW(&ctx,&conv_params,&c_quant_params,&input_dims,&section[196608],&filter_dims,weight_19,&bias_dims,bias_19,&output_dims,section,88467);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,128,256,1,section,&section[196608]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=96;

    conv_params.padding.h=0;
    conv_params.padding.w=0;

    memcpy(conv_mult_use,mult_20,384);

    for(int i = 0; i < 96; i++) { conv_shift_use[i] = shift_20[i]; }

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,128,1,&section[196608],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    arm_convolve_s8_sparse_1x1_CHW(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_20,&bias_dims,bias_20,&output_dims,&section[204800],7374);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,96,256,1,&section[204800],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    memcpy(&section[204800],section,24576);

    norm_params.input_offset=128;
    norm_params.output_offset=-87;

    t_quant_params.multiplier=2146633600;

    arm_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,96,weight_21,bias_21,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    fc_params.input_offset=87;
    fc_params.output_offset=-5;

    t_quant_params.multiplier=1438111872;
    t_quant_params.shift=-8;

    input_dims.n=256;

    filter_dims.n=96;

    output_dims.c=288;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,96,1,section,&section[180224]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    arm_fully_connected_s8_sparse_CHW(&ctx,&fc_params,&t_quant_params,&input_dims,&section[180224],&filter_dims,weight_22,&bias_dims,NULL,&output_dims,section,16590);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,288,256,1,section,&section[131072]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,6,48,&section[131072],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    memcpy(&section[131072],section,73728);

    arm_nn_batch_mat_mult_nt_t_s8(&section[131072],&section[155648],NULL,section,1769460224,-8,256,48,256,5,5,-7,2,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_softmax_s8_fast(&ctx,section,512,256,1801219200,22,-248,section);

    end = HAL_GetTick();
    softmax_time += (end - start);
    softmax_count++;
    start = end;

    arm_nn_batch_mat_mult_s8(&ctx,section,&section[180224],NULL,&section[131072],1347038848,-6,256,256,48,128,5,-13,2,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,2,256,48,&section[131072],&section[180224]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    fc_params.input_offset=13;
    fc_params.output_offset=-17;

    t_quant_params.multiplier=2105830784;
    t_quant_params.shift=-9;



    output_dims.c=96;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,96,1,&section[180224],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    arm_fully_connected_s8_sparse_CHW(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_23,&bias_dims,bias_23,&output_dims,&section[180224],5532);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,96,256,1,&section[180224],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[204800],17,2136739840,-1,128,1533058688,0,0,&section[24576],-75,2147483647,0,-128,127,24576);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[24576],24576);

    norm_params.input_offset=75;
    norm_params.output_offset=-40;

    t_quant_params.multiplier=1899635328;

    arm_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,96,weight_24,bias_24,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    memcpy(&section[204800],section,24576);

    fc_params.input_offset=40;
    fc_params.output_offset=-128;

    t_quant_params.multiplier=1996709632;
    t_quant_params.shift=-7;



    output_dims.c=128;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,96,1,section,&section[180224]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    arm_fully_connected_s8_sparse_CHW(&ctx,&fc_params,&t_quant_params,&input_dims,&section[180224],&filter_dims,weight_25,&bias_dims,bias_25,&output_dims,section,7371);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,128,256,1,section,&section[172032]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    fc_params.input_offset=128;
    fc_params.output_offset=4;

    t_quant_params.multiplier=1943961600;
    t_quant_params.shift=-8;


    filter_dims.n=128;

    output_dims.c=96;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,128,1,&section[172032],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    arm_fully_connected_s8_sparse_CHW(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_26,&bias_dims,bias_26,&output_dims,&section[180224],7371);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,96,256,1,&section[180224],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[204800],-4,1082566272,0,40,1828365312,0,0,&section[24576],-32,2147483647,0,-128,127,24576);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[24576],24576);

    input_dims.n=1;
    input_dims.c=96;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=32;

    memcpy(conv_mult_use,mult_27,384);

    for(int i = 0; i < 96; i++) { conv_shift_use[i] = shift_27[i]; }

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,96,1,section,&section[204800]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    arm_convolve_s8_sparse_1x1_CHW(&ctx,&conv_params,&c_quant_params,&input_dims,&section[204800],&filter_dims,weight_27,&bias_dims,bias_27,&output_dims,section,49767);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,96,256,1,section,&section[204800]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=256;

    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_28,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_28[i]; }

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,96,1,&section[204800],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    arm_convolve_s8_sparse_1x1_CHW(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_28,&bias_dims,bias_28,&output_dims,&section[163840],14748);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,256,1,&section[163840],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    printf("transformer0_1 finished\r\n");

    // block: last conv

    input_dims.c=256;


    output_dims.c=512;


    memcpy(conv_mult_use,mult_29,2048);

    for(int i = 0; i < 512; i++) { conv_shift_use[i] = shift_29[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_29,&bias_dims,bias_29,&output_dims,&section[98304],78636);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(section,&section[98304],131072);

    printf("last_conv finished\r\n");

    // block: qglobal_pooling

    input_dims.c=512;

    filter_dims.h=16;
    filter_dims.w=16;

    output_dims.h=1;
    output_dims.w=1;

    pool_params.stride.h=16;
    pool_params.stride.w=16;
    pool_params.padding.h=0;
    pool_params.padding.w=0;
    pool_params.activation.min=-128;
    pool_params.activation.max=127;

    arm_avgpool_s8_with_quantization(&ctx,&pool_params,&input_dims,section,&filter_dims,&output_dims,128,-128,1105721945,3,&section[228864]);

    end = HAL_GetTick();
    pool_time += (end - start);
    pool_count++;
    start = end;

    memcpy(section,&section[228864],512);

    printf("qglobal_pooling finished\r\n");

    // block: classifier

    fc_params.output_offset=-19;

    t_quant_params.multiplier=1330214144;
    t_quant_params.shift=-9;


    filter_dims.n=512;

    output_dims.c=10;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_30,&bias_dims,NULL,&output_dims,&section[229366]);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    memcpy(section,&section[229366],10);

    printf("classifier finished\r\n");

    result_check_statistics(&ctx,section,conv_count,conv_time,linear_count,linear_time,trans_count,trans_time,softmax_count,softmax_time,norm_count,norm_time,pool_count,pool_time,matmul_count,matmul_time,add_count,add_time,3);

    return 0;
}
