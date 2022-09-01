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

    static q7_t buf[2048]={0};
    static int32_t conv_mult_use[512]={0};
    static int32_t conv_shift_use[512]={0};
 
    static q7_t section[229376]={0};

    c_quant_params.multiplier=conv_mult_use;
    c_quant_params.shift=conv_shift_use;

    ctx.size = sizeof(buf);
    ctx.buf = buf;

    memcpy(&section,&image,3072);

    start = HAL_GetTick();

    // block: downsample1_0

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
    conv_params.input_offset=-2;
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
    conv_params.output_offset=-8;

    memcpy(conv_mult_use,mult_2,128);

    for(int i = 0; i < 32; i++) { conv_shift_use[i] = shift_2[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_2,&bias_dims,bias_2,&output_dims,&section[221184]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(section,&section[221184],8192);

    printf("downsample1_0 finished\r\n");

    // block: mv2block1_0

    input_dims.c=32;


    output_dims.c=96;

    conv_params.input_offset=8;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_3,384);

    for(int i = 0; i < 96; i++) { conv_shift_use[i] = shift_3[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_3,&bias_dims,bias_3,&output_dims,&section[204800],1845);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=1;

    filter_dims.h=3;
    filter_dims.w=3;


    dw_conv_params.stride.h=1;
    dw_conv_params.stride.w=1;

    memcpy(conv_mult_use,mult_4,384);

    for(int i = 0; i < 96; i++) { conv_shift_use[i] = shift_4[i]; }

    arm_depthwise_conv_s8_sparse(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[204800],&filter_dims,weight_4,&bias_dims,bias_4,&output_dims,section,799);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=96;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=64;

    conv_params.input_offset=128;
    conv_params.output_offset=-7;

    memcpy(conv_mult_use,mult_5,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_5[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_5,&bias_dims,bias_5,&output_dims,&section[212992],3687);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(section,&section[212992],16384);

    printf("mv2block1_0 finished\r\n");

    // block: mv2block2_1

    memcpy(&section[212992],section,16384);

    input_dims.c=64;


    output_dims.c=256;

    conv_params.input_offset=7;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_6,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_6[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_6,&bias_dims,bias_6,&output_dims,&section[147456],9831);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=1;

    filter_dims.h=3;
    filter_dims.w=3;



    memcpy(conv_mult_use,mult_7,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_7[i]; }

    arm_depthwise_conv_s8_sparse(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[147456],&filter_dims,weight_7,&bias_dims,bias_7,&output_dims,section,2108);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=256;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=64;

    conv_params.input_offset=128;
    conv_params.output_offset=-3;

    memcpy(conv_mult_use,mult_8,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_8[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_8,&bias_dims,bias_8,&output_dims,&section[196608],9831);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(&section[196608],&section[212992],3,1277464192,0,7,1749507328,0,0,section,-4,2147483647,0,-128,127,16384);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    printf("mv2block2_1 finished\r\n");

    // block: transformer0_0

    input_dims.c=64;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=4;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_9,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_9[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_9,&bias_dims,bias_9,&output_dims,&section[212992],16590);

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

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[212992],&filter_dims,weight_10,&bias_dims,bias_10,&output_dims,section,2766);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(&section[204800],section,24576);

    norm_params.activation.max=127;
    norm_params.activation.min=-128;
    norm_params.input_offset=128;
    norm_params.output_offset=-79;

    t_quant_params.multiplier=2034313600;
    t_quant_params.shift=-9;

    arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,96,weight_11,bias_11,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    fc_params.activation.max=127;
    fc_params.activation.min=-128;
    fc_params.input_offset=79;
    fc_params.output_offset=4;

    t_quant_params.multiplier=1469077504;
    t_quant_params.shift=-8;

    input_dims.n=256;

    filter_dims.n=96;

    output_dims.c=288;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_12,&bias_dims,NULL,&output_dims,&section[131072],12444);

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

    arm_nn_batch_mat_mult_nt_t_s8(&section[131072],&section[155648],NULL,section,1164033408,-8,256,48,256,-4,-4,-3,2,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_softmax_s8_fast(&ctx,section,512,256,1207389440,23,-248,section);

    end = HAL_GetTick();
    softmax_time += (end - start);
    softmax_count++;
    start = end;

    arm_nn_batch_mat_mult_s8(&ctx,section,&section[180224],NULL,&section[131072],1355439488,-6,256,256,48,128,-4,-4,2,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,2,256,48,&section[131072],&section[180224]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    fc_params.input_offset=4;
    fc_params.output_offset=-50;

    t_quant_params.multiplier=1941480192;
    t_quant_params.shift=-9;



    output_dims.c=96;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[180224],&filter_dims,weight_13,&bias_dims,bias_13,&output_dims,section,4149);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[204800],50,1217384064,-1,128,1721887360,0,0,&section[24576],-107,2147483647,0,-128,127,24576);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[24576],24576);

    norm_params.input_offset=107;
    norm_params.output_offset=-44;

    t_quant_params.multiplier=1840683008;

    arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,96,weight_14,bias_14,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    memcpy(&section[204800],section,24576);

    fc_params.input_offset=44;
    fc_params.output_offset=-128;

    t_quant_params.multiplier=2097448320;
    t_quant_params.shift=-7;



    output_dims.c=128;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_15,&bias_dims,bias_15,&output_dims,&section[172032],5532);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    fc_params.input_offset=128;
    fc_params.output_offset=-19;

    t_quant_params.multiplier=1207373056;
    t_quant_params.shift=-8;


    filter_dims.n=128;

    output_dims.c=96;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[172032],&filter_dims,weight_16,&bias_dims,bias_16,&output_dims,section,5532);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[204800],19,1521622016,-1,44,1898971520,0,0,&section[24576],-41,2147483647,0,-128,127,24576);

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
    conv_params.input_offset=41;

    memcpy(conv_mult_use,mult_17,384);

    for(int i = 0; i < 96; i++) { conv_shift_use[i] = shift_17[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_17,&bias_dims,bias_17,&output_dims,&section[204800],37326);

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

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[204800],&filter_dims,weight_18,&bias_dims,bias_18,&output_dims,section,5532);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    printf("transformer0_0 finished\r\n");

    // block: downsample1_0

    input_dims.c=128;


    output_dims.c=256;


    memcpy(conv_mult_use,mult_19,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_19[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_19,&bias_dims,bias_19,&output_dims,&section[163840],19662);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=1;

    filter_dims.h=3;
    filter_dims.w=3;

    output_dims.h=8;
    output_dims.w=8;

    dw_conv_params.stride.h=2;
    dw_conv_params.stride.w=2;

    memcpy(conv_mult_use,mult_20,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_20[i]; }

    arm_depthwise_conv_s8_sparse(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[163840],&filter_dims,weight_20,&bias_dims,bias_20,&output_dims,section,2100);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.h=8;
    input_dims.w=8;
    input_dims.c=256;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=128;

    conv_params.output_offset=1;

    memcpy(conv_mult_use,mult_21,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_21[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_21,&bias_dims,bias_21,&output_dims,&section[221184],19662);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(section,&section[221184],8192);

    printf("downsample1_0 finished\r\n");

    // block: mv2block2_0

    memcpy(&section[221184],section,8192);

    input_dims.c=128;


    output_dims.c=512;

    conv_params.input_offset=-1;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_22,2048);

    for(int i = 0; i < 512; i++) { conv_shift_use[i] = shift_22[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_22,&bias_dims,bias_22,&output_dims,&section[188416],39324);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=1;

    filter_dims.h=3;
    filter_dims.w=3;


    dw_conv_params.stride.h=1;
    dw_conv_params.stride.w=1;

    memcpy(conv_mult_use,mult_23,2048);

    for(int i = 0; i < 512; i++) { conv_shift_use[i] = shift_23[i]; }

    arm_depthwise_conv_s8_sparse(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[188416],&filter_dims,weight_23,&bias_dims,bias_23,&output_dims,section,4054);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=512;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=128;

    conv_params.input_offset=128;
    conv_params.output_offset=0;

    memcpy(conv_mult_use,mult_24,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_24[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_24,&bias_dims,bias_24,&output_dims,&section[212992],39324);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(&section[212992],&section[221184],0,1345889152,0,-1,1669464064,0,0,section,1,2147483647,0,-128,127,8192);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    printf("mv2block2_0 finished\r\n");

    // block: transformer1_0

    input_dims.c=128;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=-1;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_25,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_25[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_25,&bias_dims,bias_25,&output_dims,&section[221184],88476);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;


    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_26,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_26[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[221184],&filter_dims,weight_26,&bias_dims,bias_26,&output_dims,section,9831);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(&section[221184],section,8192);

    norm_params.input_offset=128;
    norm_params.output_offset=-89;

    t_quant_params.multiplier=1128445824;

    arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,64,128,weight_27,bias_27,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    fc_params.input_offset=89;
    fc_params.output_offset=-9;

    t_quant_params.multiplier=1178400256;

    input_dims.n=64;


    output_dims.c=384;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_28,&bias_dims,NULL,&output_dims,&section[196608],29493);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,64,12,32,&section[196608],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    memcpy(&section[196608],section,24576);

    arm_nn_batch_mat_mult_nt_t_s8(&section[196608],&section[204800],NULL,section,1923843072,-8,64,32,64,9,9,-9,4,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_softmax_s8_fast(&ctx,section,256,64,1649349504,22,-248,section);

    end = HAL_GetTick();
    softmax_time += (end - start);
    softmax_count++;
    start = end;

    arm_nn_batch_mat_mult_s8(&ctx,section,&section[212992],NULL,&section[16384],1113918976,-6,64,64,32,128,9,-2,4,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,4,64,32,&section[16384],&section[212992]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    fc_params.input_offset=2;
    fc_params.output_offset=-60;

    t_quant_params.multiplier=1636711808;
    t_quant_params.shift=-9;



    output_dims.c=128;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[212992],&filter_dims,weight_29,&bias_dims,bias_29,&output_dims,section,9831);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[221184],60,1254776192,0,128,1375181056,0,0,&section[8192],-92,2147483647,0,-128,127,8192);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[8192],8192);

    norm_params.input_offset=92;
    norm_params.output_offset=-52;

    t_quant_params.multiplier=1131436544;
    t_quant_params.shift=-8;

    arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,64,128,weight_30,bias_30,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    memcpy(&section[221184],section,8192);

    fc_params.input_offset=52;
    fc_params.output_offset=-128;

    t_quant_params.multiplier=1470327680;
    t_quant_params.shift=-7;



    output_dims.c=256;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_31,&bias_dims,bias_31,&output_dims,&section[204800],19662);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    fc_params.input_offset=128;
    fc_params.output_offset=2;

    t_quant_params.multiplier=1512711040;
    t_quant_params.shift=-9;


    filter_dims.n=256;

    output_dims.c=128;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[204800],&filter_dims,weight_32,&bias_dims,bias_32,&output_dims,section,19662);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[221184],-2,2044380416,-1,52,1883044224,0,0,&section[8192],-39,2147483647,0,-128,127,8192);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[8192],8192);

    input_dims.n=1;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=39;

    memcpy(conv_mult_use,mult_33,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_33[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_33,&bias_dims,bias_33,&output_dims,&section[221184],88476);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=192;

    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_34,768);

    for(int i = 0; i < 192; i++) { conv_shift_use[i] = shift_34[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[221184],&filter_dims,weight_34,&bias_dims,bias_34,&output_dims,section,14748);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    printf("transformer1_0 finished\r\n");

    // block: transformer1_1

    input_dims.c=192;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;

    memcpy(conv_mult_use,mult_35,768);

    for(int i = 0; i < 192; i++) { conv_shift_use[i] = shift_35[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_35,&bias_dims,bias_35,&output_dims,&section[217088],149298);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=128;

    conv_params.padding.h=0;
    conv_params.padding.w=0;

    memcpy(conv_mult_use,mult_36,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_36[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[217088],&filter_dims,weight_36,&bias_dims,bias_36,&output_dims,section,11061);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(&section[221184],section,8192);

    norm_params.input_offset=128;
    norm_params.output_offset=-90;

    t_quant_params.multiplier=1164345984;
    t_quant_params.shift=-8;

    arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,64,128,weight_37,bias_37,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    fc_params.input_offset=90;
    fc_params.output_offset=0;

    t_quant_params.multiplier=1297318912;

    input_dims.n=64;

    filter_dims.n=128;

    output_dims.c=384;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_38,&bias_dims,NULL,&output_dims,&section[196608],22119);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,64,12,32,&section[196608],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    memcpy(&section[196608],section,24576);

    arm_nn_batch_mat_mult_nt_t_s8(&section[196608],&section[204800],NULL,section,1939239680,-8,64,32,64,0,0,7,4,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_softmax_s8_fast(&ctx,section,256,64,1713889152,22,-248,section);

    end = HAL_GetTick();
    softmax_time += (end - start);
    softmax_count++;
    start = end;

    arm_nn_batch_mat_mult_s8(&ctx,section,&section[212992],NULL,&section[16384],1902573568,-7,64,64,32,128,0,-2,4,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,4,64,32,&section[16384],&section[212992]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    fc_params.input_offset=2;
    fc_params.output_offset=-15;

    t_quant_params.multiplier=1985354752;
    t_quant_params.shift=-9;



    output_dims.c=128;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[212992],&filter_dims,weight_39,&bias_dims,bias_39,&output_dims,section,7374);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[221184],15,2131141248,-1,128,1596438272,0,0,&section[8192],-76,2147483647,0,-128,127,8192);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[8192],8192);

    norm_params.input_offset=76;
    norm_params.output_offset=-41;

    t_quant_params.multiplier=1934036736;

    arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,64,128,weight_40,bias_40,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    memcpy(&section[221184],section,8192);

    fc_params.input_offset=41;
    fc_params.output_offset=-128;

    t_quant_params.multiplier=1600355584;
    t_quant_params.shift=-7;



    output_dims.c=256;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_41,&bias_dims,bias_41,&output_dims,&section[204800],14748);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    fc_params.input_offset=128;
    fc_params.output_offset=-23;

    t_quant_params.multiplier=1669974528;
    t_quant_params.shift=-9;


    filter_dims.n=256;

    output_dims.c=128;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[204800],&filter_dims,weight_42,&bias_dims,bias_42,&output_dims,section,14748);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[221184],23,1336785152,0,41,1721721088,0,0,&section[8192],-40,2147483647,0,-128,127,8192);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[8192],8192);

    input_dims.n=1;
    input_dims.c=128;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=40;

    memcpy(conv_mult_use,mult_43,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_43[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_43,&bias_dims,bias_43,&output_dims,&section[221184],66357);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=256;

    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_44,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_44[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[221184],&filter_dims,weight_44,&bias_dims,bias_44,&output_dims,section,14748);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    printf("transformer1_1 finished\r\n");

    // block: last conv

    input_dims.c=256;


    output_dims.c=512;


    memcpy(conv_mult_use,mult_45,2048);

    for(int i = 0; i < 512; i++) { conv_shift_use[i] = shift_45[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_45,&bias_dims,bias_45,&output_dims,&section[196608],98292);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(section,&section[196608],32768);

    printf("last_conv finished\r\n");

    // block: qglobal_pooling

    input_dims.c=512;

    filter_dims.h=8;
    filter_dims.w=8;

    output_dims.h=1;
    output_dims.w=1;

    pool_params.stride.h=8;
    pool_params.stride.w=8;
    pool_params.padding.h=0;
    pool_params.padding.w=0;
    pool_params.activation.min=-128;
    pool_params.activation.max=127;

    arm_avgpool_s8_with_quantization(&ctx,&pool_params,&input_dims,section,&filter_dims,&output_dims,128,-128,1420100117,3,&section[228864]);

    end = HAL_GetTick();
    pool_time += (end - start);
    pool_count++;
    start = end;

    memcpy(section,&section[228864],512);

    printf("qglobal_pooling finished\r\n");

    // block: classifier

    fc_params.output_offset=-4;

    t_quant_params.multiplier=1274933248;
    t_quant_params.shift=-8;


    filter_dims.n=512;

    output_dims.c=10;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_46,&bias_dims,NULL,&output_dims,&section[229366]);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    memcpy(section,&section[229366],10);

    printf("classifier finished\r\n");

    result_check_statistics(&ctx,section,conv_count,conv_time,linear_count,linear_time,trans_count,trans_time,softmax_count,softmax_time,norm_count,norm_time,pool_count,pool_time,matmul_count,matmul_time,add_count,add_time,3);

    return 0;
}
