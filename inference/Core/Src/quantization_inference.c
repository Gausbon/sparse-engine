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

    // block: downsample0_0

    input_dims.n=1;
    input_dims.h=32;
    input_dims.w=32;
    input_dims.c=3;

    filter_dims.h=3;
    filter_dims.w=3;

    output_dims.h=32;
    output_dims.w=32;
    output_dims.c=64;

    conv_params.stride.h=1;
    conv_params.stride.w=1;
    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.activation.max=127;
    conv_params.activation.min=-128;
    conv_params.input_offset=-2;
    conv_params.output_offset=-128;
    conv_params.dilation.h=1;
    conv_params.dilation.w=1;

    memcpy(conv_mult_use,mult_0,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_0[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_0,&bias_dims,NULL,&output_dims,&section[131072]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=64;


    output_dims.h=16;
    output_dims.w=16;

    pool_params.stride.h=2;
    pool_params.stride.w=2;
    pool_params.padding.h=1;
    pool_params.padding.w=1;
    pool_params.activation.min=-128;
    pool_params.activation.max=127;

    arm_maxpool_s8_with_quantization(&ctx,&pool_params,&input_dims,&section[131072],&filter_dims,&output_dims,128,-128,1073741824,1,section);

    end = HAL_GetTick();
    pool_time += (end - start);
    pool_count++;
    start = end;

    printf("downsample0_0 finished\r\n");

    // block: mv2block1_0

    memcpy(&section[180224],section,16384);

    input_dims.h=16;
    input_dims.w=16;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=192;

    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_1,768);

    for(int i = 0; i < 192; i++) { conv_shift_use[i] = shift_1[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_1,&bias_dims,bias_1,&output_dims,&section[131072],7374);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=1;

    filter_dims.h=3;
    filter_dims.w=3;


    dw_conv_params.stride.h=1;
    dw_conv_params.stride.w=1;
    dw_conv_params.padding.h=1;
    dw_conv_params.padding.w=1;
    dw_conv_params.activation.max=127;
    dw_conv_params.activation.min=-128;
    dw_conv_params.input_offset=128;
    dw_conv_params.output_offset=-128;
    dw_conv_params.dilation.h=1;
    dw_conv_params.dilation.w=1;
    dw_conv_params.ch_mult=1;

    memcpy(conv_mult_use,mult_2,768);

    for(int i = 0; i < 192; i++) { conv_shift_use[i] = shift_2[i]; }

    arm_depthwise_conv_s8_sparse(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[131072],&filter_dims,weight_2,&bias_dims,bias_2,&output_dims,section,924);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=192;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=64;

    conv_params.output_offset=-4;

    memcpy(conv_mult_use,mult_3,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_3[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_3,&bias_dims,bias_3,&output_dims,&section[163840],7374);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(&section[163840],&section[180224],4,1075795968,1,128,1910288000,-2,0,section,-16,2147483647,0,-128,127,16384);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    printf("mv2block1_0 finished\r\n");

    // block: transformer0_0

    input_dims.c=64;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=16;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_4,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_4[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_4,&bias_dims,bias_4,&output_dims,&section[180224],33177);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;


    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_5,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_5[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[180224],&filter_dims,weight_5,&bias_dims,bias_5,&output_dims,section,3687);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(&section[180224],section,16384);

    norm_params.activation.max=127;
    norm_params.activation.min=-128;
    norm_params.input_offset=128;
    norm_params.output_offset=-79;

    t_quant_params.multiplier=1368220672;
    t_quant_params.shift=-8;

    arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,64,weight_6,bias_6,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    fc_params.activation.max=127;
    fc_params.activation.min=-128;
    fc_params.input_offset=79;
    fc_params.output_offset=10;

    t_quant_params.multiplier=1311623936;

    input_dims.n=256;

    filter_dims.n=64;

    output_dims.c=192;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_7,&bias_dims,NULL,&output_dims,&section[131072],11061);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_nn_transpose_bnc_to_nbc_q7(256,6,32,&section[131072],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    memcpy(&section[131072],section,49152);

    arm_nn_batch_mat_mult_nt_t_s8(&section[131072],&section[147456],NULL,section,1770372224,-8,256,32,256,-10,-10,-9,2,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_softmax_s8(section,512,256,1836800896,22,-248,section);

    end = HAL_GetTick();
    softmax_time += (end - start);
    softmax_count++;
    start = end;

    arm_nn_batch_mat_mult_s8(&ctx,section,&section[163840],NULL,&section[131072],1102211584,-6,256,256,32,128,-10,0,2,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_nn_transpose_bnc_to_nbc_q7(2,256,32,&section[131072],&section[163840]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    fc_params.input_offset=0;
    fc_params.output_offset=-2;

    t_quant_params.multiplier=1350139008;



    output_dims.c=64;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[163840],&filter_dims,weight_8,&bias_dims,bias_8,&output_dims,section,3687);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[180224],2,1399134848,-1,128,1687859328,0,0,&section[16384],-92,2147483647,0,-128,127,16384);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[16384],16384);

    norm_params.input_offset=92;
    norm_params.output_offset=-32;

    t_quant_params.multiplier=2094177024;
    t_quant_params.shift=-9;

    arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,64,weight_9,bias_9,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    memcpy(&section[180224],section,16384);

    fc_params.input_offset=32;
    fc_params.output_offset=-128;

    t_quant_params.multiplier=1697654784;
    t_quant_params.shift=-6;



    output_dims.c=512;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_10,&bias_dims,bias_10,&output_dims,&section[49152],29490);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    fc_params.input_offset=128;
    fc_params.output_offset=-19;

    t_quant_params.multiplier=1124316416;
    t_quant_params.shift=-9;


    filter_dims.n=512;

    output_dims.c=64;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[49152],&filter_dims,weight_11,&bias_dims,bias_11,&output_dims,section,29493);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[180224],19,1301784832,0,32,1967344896,0,0,&section[16384],-34,2147483647,0,-128,127,16384);

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
    conv_params.input_offset=34;

    memcpy(conv_mult_use,mult_12,256);

    for(int i = 0; i < 64; i++) { conv_shift_use[i] = shift_12[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_12,&bias_dims,bias_12,&output_dims,&section[180224],33177);

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

    memcpy(conv_mult_use,mult_13,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_13[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[180224],&filter_dims,weight_13,&bias_dims,bias_13,&output_dims,section,7374);

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

    memcpy(conv_mult_use,mult_14,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_14[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_14,&bias_dims,NULL,&output_dims,&section[163840]);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;



    output_dims.h=8;
    output_dims.w=8;


    arm_maxpool_s8_with_quantization(&ctx,&pool_params,&input_dims,&section[163840],&filter_dims,&output_dims,128,-128,1073741824,1,section);

    end = HAL_GetTick();
    pool_time += (end - start);
    pool_count++;
    start = end;

    printf("downsample0_0 finished\r\n");

    // block: mv2block0_0

    memcpy(&section[188416],section,8192);

    input_dims.h=8;
    input_dims.w=8;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=256;

    conv_params.padding.h=0;
    conv_params.padding.w=0;

    memcpy(conv_mult_use,mult_15,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_15[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_15,&bias_dims,bias_15,&output_dims,&section[172032],29493);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=1;

    filter_dims.h=3;
    filter_dims.w=3;



    memcpy(conv_mult_use,mult_16,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_16[i]; }

    arm_depthwise_conv_s8_sparse(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[172032],&filter_dims,weight_16,&bias_dims,bias_16,&output_dims,section,1844);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    input_dims.c=256;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=128;

    conv_params.output_offset=0;

    memcpy(conv_mult_use,mult_17,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_17[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_17,&bias_dims,bias_17,&output_dims,&section[180224],29493);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(&section[180224],&section[188416],0,1553458048,0,128,1196041088,0,0,section,-44,2147483647,0,-128,127,8192);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    printf("mv2block0_0 finished\r\n");

    // block: transformer0_0

    input_dims.c=128;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=44;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_18,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_18[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_18,&bias_dims,bias_18,&output_dims,&section[188416],66354);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;


    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_19,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_19[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[188416],&filter_dims,weight_19,&bias_dims,bias_19,&output_dims,section,7374);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(&section[188416],section,8192);

    norm_params.input_offset=128;
    norm_params.output_offset=-89;

    t_quant_params.multiplier=1092726656;
    t_quant_params.shift=-8;

    arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,64,128,weight_20,bias_20,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    fc_params.input_offset=89;
    fc_params.output_offset=-4;

    t_quant_params.multiplier=1637402368;

    input_dims.n=64;

    filter_dims.n=128;

    output_dims.c=384;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_21,&bias_dims,NULL,&output_dims,&section[163840],22119);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_nn_transpose_bnc_to_nbc_q7(64,6,64,&section[163840],section);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    memcpy(&section[163840],section,24576);

    arm_nn_batch_mat_mult_nt_t_s8(&section[163840],&section[172032],NULL,section,1371455488,-8,64,64,64,4,4,-14,2,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_softmax_s8(section,128,64,1470711936,22,-248,section);

    end = HAL_GetTick();
    softmax_time += (end - start);
    softmax_count++;
    start = end;

    arm_nn_batch_mat_mult_s8(&ctx,section,&section[180224],NULL,&section[8192],1126557440,-6,64,64,64,128,4,-6,2,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    arm_nn_transpose_bnc_to_nbc_q7(2,64,64,&section[8192],&section[180224]);

    end = HAL_GetTick();
    trans_time += (end - start);
    trans_count++;
    start = end;

    fc_params.input_offset=6;
    fc_params.output_offset=-31;

    t_quant_params.multiplier=1160828544;



    output_dims.c=128;

    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[180224],&filter_dims,weight_22,&bias_dims,bias_22,&output_dims,section,7374);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[188416],31,1871851264,-1,128,1853789056,0,0,&section[8192],-88,2147483647,0,-128,127,8192);

    end = HAL_GetTick();
    add_time += (end - start);
    add_count++;
    start = end;

    memcpy(section,&section[8192],8192);

    norm_params.input_offset=88;
    norm_params.output_offset=-57;

    t_quant_params.multiplier=1182694400;

    arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,64,128,weight_23,bias_23,section,section);

    end = HAL_GetTick();
    norm_time += (end - start);
    norm_count++;
    start = end;

    memcpy(&section[188416],section,8192);

    fc_params.input_offset=57;
    fc_params.output_offset=-128;

    t_quant_params.multiplier=1110894848;
    t_quant_params.shift=-6;




    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_24,&bias_dims,bias_24,&output_dims,&section[180224],7374);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    fc_params.input_offset=128;
    fc_params.output_offset=-20;

    t_quant_params.multiplier=1202074496;
    t_quant_params.shift=-8;




    arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[180224],&filter_dims,weight_25,&bias_dims,bias_25,&output_dims,section,7374);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[188416],20,1137511552,0,57,1988968064,0,0,&section[8192],-54,2147483647,0,-128,127,8192);

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
    conv_params.input_offset=54;

    memcpy(conv_mult_use,mult_26,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_26[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_26,&bias_dims,bias_26,&output_dims,&section[188416],66357);

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

    memcpy(conv_mult_use,mult_27,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_27[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[188416],&filter_dims,weight_27,&bias_dims,bias_27,&output_dims,section,14748);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    printf("transformer0_0 finished\r\n");

    // block: last conv

    input_dims.c=256;


    output_dims.c=512;


    memcpy(conv_mult_use,mult_28,2048);

    for(int i = 0; i < 512; i++) { conv_shift_use[i] = shift_28[i]; }

    arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_28,&bias_dims,bias_28,&output_dims,&section[163840],78642);

    end = HAL_GetTick();
    conv_time += (end - start);
    conv_count++;
    start = end;

    memcpy(section,&section[163840],32768);

    printf("last_conv finished\r\n");

    // block: global_pooling

    memcpy(&section[163840],section,32768);

    fc_params.output_offset=64;

    t_quant_params.multiplier=1148889600;
    t_quant_params.shift=-9;

    input_dims.n=64;

    filter_dims.n=512;

    output_dims.c=1;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_29,&bias_dims,bias_29,&output_dims,&section[163776]);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    arm_softmax_s8(&section[163776],1,64,1741366528,22,-248,&section[163776]);

    end = HAL_GetTick();
    softmax_time += (end - start);
    softmax_count++;
    start = end;

    arm_nn_batch_mat_mult_s8(&ctx,&section[163776],&section[163840],NULL,section,1276463872,-6,1,64,512,128,128,-128,1,-128,127);

    end = HAL_GetTick();
    matmul_time += (end - start);
    matmul_count++;
    start = end;

    printf("global_pooling finished\r\n");

    // block: classifier

    fc_params.output_offset=-9;

    t_quant_params.multiplier=1377633024;

    input_dims.n=1;


    output_dims.c=10;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_30,&bias_dims,NULL,&output_dims,&section[196598]);

    end = HAL_GetTick();
    linear_time += (end - start);
    linear_count++;
    start = end;

    memcpy(section,&section[196598],10);

    printf("classifier finished\r\n");

    result_check_statistics(&ctx,section,conv_count,conv_time,linear_count,linear_time,trans_count,trans_time,softmax_count,softmax_time,norm_count,norm_time,pool_count,pool_time,matmul_count,matmul_time,add_count,add_time,9);

    return 0;
}
