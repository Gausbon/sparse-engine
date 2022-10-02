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

    static q7_t buf[4608]={0};
    static int32_t conv_mult_use[512]={0};
    static int32_t conv_shift_use[512]={0};
 
    static q7_t section[196608]={0};

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

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_0,&bias_dims,bias_0,&output_dims,&section[190464]);

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

    arm_depthwise_conv_s8(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[190464],&filter_dims,weight_1,&bias_dims,bias_1,&output_dims,section);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.h=16;
    input_dims.w=16;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=64;

    conv_params.input_offset=128;
    conv_params.output_offset=0;

    memcpy(conv_mult_use,mult_2,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_2[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_2,&bias_dims,bias_2,&output_dims,&section[180224]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(section,&section[180224],16384);

    printf("downsample1_0 finished\r\n");

    // block: mv2block0_0

    memcpy(&section[180224],section,16384);

    input_dims.c=64;


    output_dims.c=128;

    conv_params.input_offset=0;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_3,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_3[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_3,&bias_dims,bias_3,&output_dims,&section[147456]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=1;

    filter_dims.h=3;
    filter_dims.w=3;


    dw_conv_params.stride.h=1;
    dw_conv_params.stride.w=1;

    memcpy(conv_mult_use,mult_4,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_4[i]; }

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,128,1,&section[147456],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    arm_depthwise_conv_s8_sparse_1x1_CHW(&ctx,&dw_conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_4,&bias_dims,bias_4,&output_dims,&section[147456],1076);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,128,256,1,&section[147456],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    input_dims.c=128;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=64;

    conv_params.input_offset=128;
    conv_params.output_offset=-5;

    memcpy(conv_mult_use,mult_5,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_5[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_5,&bias_dims,bias_5,&output_dims,&section[163840]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(&section[163840],&section[180224],5,1621124608,0,0,1580184704,0,0,section,-10,2147483647,0,-128,127,16384);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    printf("mv2block0_0 finished\r\n");

    // block: transformer0_0

    input_dims.c=64;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=10;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_6,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_6[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_6,&bias_dims,bias_6,&output_dims,&section[180224]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;


    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_7,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_7[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,&section[180224],&filter_dims,weight_7,&bias_dims,bias_7,&output_dims,section);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(&section[180224],section,16384);

    norm_params.activation.max=127;
    norm_params.activation.min=-128;
    norm_params.input_offset=128;
    norm_params.output_offset=-77;

    t_quant_params.multiplier=1187804800;
    t_quant_params.shift=-8;

    arm_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,64,weight_8,bias_8,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    fc_params.activation.max=127;
    fc_params.activation.min=-128;
    fc_params.input_offset=77;
    fc_params.output_offset=8;

    t_quant_params.multiplier=1426356608;

    input_dims.n=256;

    filter_dims.n=64;

    output_dims.c=192;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_9,&bias_dims,NULL,&output_dims,&section[131072]);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,6,32,&section[131072],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    memcpy(&section[131072],section,49152);

    arm_nn_batch_mat_mult_nt_t_s8(&section[131072],&section[147456],NULL,section,1472306176,-8,256,32,256,-8,-8,-8,2,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_softmax_s8_fast(&ctx,section,512,256,1125339776,23,-248,section);

    end = HAL_GetTick();
    softmax_time += (end - start);
    softmax_count++;
    start = end;

    arm_nn_batch_mat_mult_s8(&ctx,section,&section[163840],NULL,&section[131072],1259360640,-6,256,256,32,128,-8,-4,2,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,2,256,32,&section[131072],&section[163840]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    fc_params.input_offset=4;
    fc_params.output_offset=-28;

    t_quant_params.multiplier=2025732480;
    t_quant_params.shift=-9;



    output_dims.c=64;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,&section[163840],&filter_dims,weight_10,&bias_dims,bias_10,&output_dims,section);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[180224],28,2112488320,-2,128,1783450368,0,0,&section[16384],-107,2147483647,0,-128,127,16384);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[16384],16384);

    norm_params.input_offset=107;
    norm_params.output_offset=-30;

    t_quant_params.multiplier=1908797184;

    arm_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,64,weight_11,bias_11,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    memcpy(&section[180224],section,16384);

    fc_params.input_offset=30;
    fc_params.output_offset=-128;

    t_quant_params.multiplier=1443283072;
    t_quant_params.shift=-6;



    output_dims.c=512;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_12,&bias_dims,bias_12,&output_dims,&section[49152]);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    fc_params.input_offset=128;
    fc_params.output_offset=-8;

    t_quant_params.multiplier=1098378112;
    t_quant_params.shift=-9;


    filter_dims.n=512;

    output_dims.c=64;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,&section[49152],&filter_dims,weight_13,&bias_dims,bias_13,&output_dims,section);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[180224],8,1315054720,0,30,1899676288,0,0,&section[16384],-27,2147483647,0,-128,127,16384);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[16384],16384);

    input_dims.n=1;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=27;

    memcpy(conv_mult_use,mult_14,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_14[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_14,&bias_dims,bias_14,&output_dims,&section[180224]);

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

    memcpy(conv_mult_use,mult_15,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_15[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,&section[180224],&filter_dims,weight_15,&bias_dims,bias_15,&output_dims,section);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    printf("transformer0_0 finished\r\n");

    // block: downsample0_0

    input_dims.c=128;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;

    memcpy(conv_mult_use,mult_16,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_16[i]; }

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,128,1,section,&section[163840]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    arm_convolve_s8_sparse_1x1_CHW(&ctx,&conv_params,&c_quant_params,&input_dims,&section[163840],&filter_dims,weight_16,&bias_dims,NULL,&output_dims,section,55308,4);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,128,256,1,section,&section[163840]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;



    output_dims.h=8;
    output_dims.w=8;

    pool_params.stride.h=2;
    pool_params.stride.w=2;
    pool_params.padding.h=1;
    pool_params.padding.w=1;
    pool_params.activation.min=-128;
    pool_params.activation.max=127;

    arm_maxpool_s8_with_quantization(&ctx,&pool_params,&input_dims,&section[163840],&filter_dims,&output_dims,128,-128,1073741824,1,section);

    end = HAL_GetTick();
    pool_time += (end - start);
    pool_count++;
    start = end;

    printf("downsample0_0 finished\r\n");

    // block: mv2block2_0

    memcpy(&section[188416],section,8192);

    input_dims.h=8;
    input_dims.w=8;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=512;

    conv_params.padding.h=0;
    conv_params.padding.w=0;

    memcpy(conv_mult_use,mult_17,2048);

    for(int i = 0; i < 512; i++) { conv_shift_use[i] = shift_17[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_17,&bias_dims,bias_17,&output_dims,&section[155648]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=512;

    filter_dims.h=3;
    filter_dims.w=3;



    memcpy(conv_mult_use,mult_18,2048);

    for(int i = 0; i < 512; i++) { conv_shift_use[i] = shift_18[i]; }

    arm_depthwise_conv_s8(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[155648],&filter_dims,weight_18,&bias_dims,bias_18,&output_dims,section);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=128;

    conv_params.output_offset=6;

    memcpy(conv_mult_use,mult_19,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_19[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_19,&bias_dims,bias_19,&output_dims,&section[180224]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(&section[180224],&section[188416],-6,1658686720,0,128,1100768768,0,0,section,-35,2147483647,0,-128,127,8192);

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
    conv_params.input_offset=35;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_20,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_20[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_20,&bias_dims,bias_20,&output_dims,&section[188416]);

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

    memcpy(conv_mult_use,mult_21,384);

    for(int i = 0; i < 96; i++) { conv_shift_use[i] = shift_21[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,&section[188416],&filter_dims,weight_21,&bias_dims,bias_21,&output_dims,section);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(&section[190464],section,6144);

    norm_params.input_offset=128;
    norm_params.output_offset=-84;

    t_quant_params.multiplier=1250342144;
    t_quant_params.shift=-8;

    arm_layernorm_s8(&ctx,&norm_params,&t_quant_params,64,96,weight_22,bias_22,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    fc_params.input_offset=84;
    fc_params.output_offset=-5;

    t_quant_params.multiplier=1250548224;

    input_dims.n=64;

    filter_dims.n=96;

    output_dims.c=288;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_23,&bias_dims,NULL,&output_dims,&section[172032]);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,64,12,24,&section[172032],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    memcpy(&section[172032],section,18432);

    arm_nn_batch_mat_mult_nt_t_s8(&section[172032],&section[178176],NULL,section,1207024512,-7,64,24,64,5,5,-5,4,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_softmax_s8_fast(&ctx,section,256,64,1519631872,22,-248,section);

    end = HAL_GetTick();
    softmax_time += (end - start);
    softmax_count++;
    start = end;

    arm_nn_batch_mat_mult_s8(&ctx,section,&section[184320],NULL,&section[16384],1159971584,-6,64,64,24,128,5,8,4,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,4,64,24,&section[16384],&section[184320]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    fc_params.input_offset=-8;
    fc_params.output_offset=-30;

    t_quant_params.multiplier=1667606528;
    t_quant_params.shift=-9;



    output_dims.c=96;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,&section[184320],&filter_dims,weight_24,&bias_dims,bias_24,&output_dims,section);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[190464],30,1159091200,0,128,1516040064,0,0,&section[6144],-77,2147483647,0,-128,127,6144);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[6144],6144);

    norm_params.input_offset=77;
    norm_params.output_offset=-43;

    t_quant_params.multiplier=1140384384;
    t_quant_params.shift=-8;

    arm_layernorm_s8(&ctx,&norm_params,&t_quant_params,64,96,weight_25,bias_25,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    memcpy(&section[190464],section,6144);

    fc_params.input_offset=43;
    fc_params.output_offset=-128;

    t_quant_params.multiplier=1729363328;
    t_quant_params.shift=-7;



    output_dims.c=128;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_26,&bias_dims,bias_26,&output_dims,&section[182272]);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    fc_params.input_offset=128;
    fc_params.output_offset=-13;

    t_quant_params.multiplier=1104272896;
    t_quant_params.shift=-8;


    filter_dims.n=128;

    output_dims.c=96;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,&section[182272],&filter_dims,weight_27,&bias_dims,bias_27,&output_dims,section);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[190464],13,1091443328,0,43,2013795584,0,0,&section[6144],-46,2147483647,0,-128,127,6144);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[6144],6144);

    input_dims.n=1;
    input_dims.c=96;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=46;

    memcpy(conv_mult_use,mult_28,384);

    for(int i = 0; i < 96; i++) { conv_shift_use[i] = shift_28[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_28,&bias_dims,bias_28,&output_dims,&section[190464]);

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

    memcpy(conv_mult_use,mult_29,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_29[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,&section[190464],&filter_dims,weight_29,&bias_dims,bias_29,&output_dims,section);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    printf("transformer1_0 finished\r\n");

    // block: last conv

    input_dims.c=256;


    output_dims.c=512;


    memcpy(conv_mult_use,mult_30,2048);

    for(int i = 0; i < 512; i++) { conv_shift_use[i] = shift_30[i]; }

    arm_nn_transpose_bhwc_to_bwhc_q7(1,64,256,1,section,&section[180224]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    arm_convolve_s8_sparse_1x1_CHW(&ctx,&conv_params,&c_quant_params,&input_dims,&section[180224],&filter_dims,weight_30,&bias_dims,bias_30,&output_dims,section,65540,4);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,512,64,1,section,&section[163840]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    memcpy(section,&section[163840],32768);

    printf("last_conv finished\r\n");

    // block: global_pooling

    memcpy(&section[163840],section,32768);

    fc_params.output_offset=82;

    t_quant_params.multiplier=1614018304;
    t_quant_params.shift=-10;

    input_dims.n=64;

    filter_dims.n=512;

    output_dims.c=1;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_31,&bias_dims,bias_31,&output_dims,&section[163776]);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_softmax_s8_fast(&ctx,&section[163776],1,64,1790261376,22,-248,&section[163776]);

    end = HAL_GetTick();
    softmax_time += (end - start);
    softmax_count++;
    start = end;

    arm_nn_batch_mat_mult_s8(&ctx,&section[163776],&section[163840],NULL,section,1233759488,-6,1,64,512,128,128,-128,1,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    printf("global_pooling finished\r\n");

    // block: classifier

    fc_params.output_offset=-11;

    t_quant_params.multiplier=1471475456;
    t_quant_params.shift=-9;

    input_dims.n=1;


    output_dims.c=10;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_32,&bias_dims,NULL,&output_dims,&section[196598]);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    memcpy(section,&section[196598],10);

    printf("classifier finished\r\n");

    result_check_statistics(&ctx,section,conv_count,conv_time,linear_count,linear_time,trans_count,trans_time,softmax_count,softmax_time,norm_count,norm_time,pool_count,pool_time,matmul_count,matmul_time,add_count,add_time,3);

    return 0;
}
