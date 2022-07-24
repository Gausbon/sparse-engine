#include "arm_nnfunctions.h"
#include "data.h"
#include "func.h"

int quantization_inference(void) {
    static q7_t buf[4*32*32]={0};
    cmsis_nn_dims input_dims, output_dims, filter_dims, bias_dims;
    
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_conv_params  conv_params;
    cmsis_nn_fc_params fc_params;
	  cmsis_nn_pool_params pool_params;
    cmsis_nn_layernorm_params norm_params;
    
    cmsis_nn_per_tensor_quant_params t_quant_params;
    cmsis_nn_per_channel_quant_params c_quant_params;

    cmsis_nn_context ctx;
  	ctx.size = sizeof(buf);
    ctx.buf = buf;
	  static int32_t conv_mult_use[512]={0};

	  static int32_t conv_shift_use[512]={0};

	  c_quant_params.multiplier=conv_mult_use;

	  c_quant_params.shift=conv_shift_use;

	  static q7_t section[307200]={0};

	  memcpy(&section,&image,3072);

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

	  memcpy(conv_mult_use,mult_0,24);

	  memcpy(conv_shift_use,shift_0,24);

	  arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_0,&bias_dims,bias_0,&output_dims,&section[307190]);

	  input_dims.c=1;

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
	  dw_conv_params.ch_mult=1;

	  memcpy(conv_mult_use,mult_1,24);

	  memcpy(conv_shift_use,shift_1,24);

	  arm_depthwise_conv_s8(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[307190],&filter_dims,weight_1,&bias_dims,bias_1,&output_dims,section);

	  input_dims.h=16;
	  input_dims.w=16;
	  input_dims.c=6;

	  filter_dims.h=1;
	  filter_dims.w=1;

	  output_dims.c=32;

	  conv_params.input_offset=128;
	  conv_params.output_offset=-7;

	  memcpy(conv_mult_use,mult_2,128);

	  memcpy(conv_shift_use,shift_2,128);

	  arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_2,&bias_dims,bias_2,&output_dims,&section[241664]);

	  memcpy(section,&section[241664],65536);

	  input_dims.c=32;


	  output_dims.c=64;

	  conv_params.input_offset=7;
	  conv_params.output_offset=-128;

	  memcpy(conv_mult_use,mult_3,256);

	  memcpy(conv_shift_use,shift_3,256);

	  arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_3,&bias_dims,bias_3,&output_dims,&section[282624]);

	  input_dims.c=1;

	  filter_dims.h=3;
	  filter_dims.w=3;


	  dw_conv_params.stride.h=1;
	  dw_conv_params.stride.w=1;

	  memcpy(conv_mult_use,mult_4,256);

	  memcpy(conv_shift_use,shift_4,256);

	  arm_depthwise_conv_s8(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[282624],&filter_dims,weight_4,&bias_dims,bias_4,&output_dims,section);

	  input_dims.c=64;

	  filter_dims.h=1;
	  filter_dims.w=1;


	  conv_params.input_offset=128;
	  conv_params.output_offset=3;

	  memcpy(conv_mult_use,mult_5,256);

	  memcpy(conv_shift_use,shift_5,256);

	  arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_5,&bias_dims,bias_5,&output_dims,&section[282624]);

	  memcpy(section,&section[282624],24576);



	  output_dims.c=128;

	  conv_params.input_offset=-3;
	  conv_params.output_offset=-128;

	  memcpy(conv_mult_use,mult_6,512);

	  memcpy(conv_shift_use,shift_6,512);

	  arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_6,&bias_dims,bias_6,&output_dims,&section[282624]);

	  input_dims.c=1;

	  filter_dims.h=3;
	  filter_dims.w=3;



	  memcpy(conv_mult_use,mult_7,512);

	  memcpy(conv_shift_use,shift_7,512);

	  arm_depthwise_conv_s8(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[282624],&filter_dims,weight_7,&bias_dims,bias_7,&output_dims,section);

	  input_dims.c=128;

	  filter_dims.h=1;
	  filter_dims.w=1;

	  output_dims.c=64;

	  conv_params.input_offset=128;
	  conv_params.output_offset=3;

	  memcpy(conv_mult_use,mult_8,256);

	  memcpy(conv_shift_use,shift_8,256);

	  arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_8,&bias_dims,bias_8,&output_dims,&section[307200]);

	  memcpy(section,&section[307200],0);

	  input_dims.c=64;

	  filter_dims.h=3;
	  filter_dims.w=3;


	  conv_params.padding.h=1;
	  conv_params.padding.w=1;
	  conv_params.input_offset=-1;
	  conv_params.output_offset=-128;

	  memcpy(conv_mult_use,mult_9,256);

	  memcpy(conv_shift_use,shift_9,256);

	  arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,&section[282624],&filter_dims,weight_9,&bias_dims,bias_9,&output_dims,section);


	  filter_dims.h=1;
	  filter_dims.w=1;

	  output_dims.c=96;

	  conv_params.padding.h=0;
	  conv_params.padding.w=0;
	  conv_params.input_offset=128;

	  memcpy(conv_mult_use,mult_10,384);

	  memcpy(conv_shift_use,shift_10,384);

	  arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_10,&bias_dims,bias_10,&output_dims,&section[282624]);

	  memcpy(&section[282624],section,24576);

	  norm_params.activation.max=127;
	  norm_params.activation.min=-128;
	  norm_params.input_offset=128;
	  norm_params.output_offset=-81;

	  t_quant_params.multiplier=1237536512;
	  t_quant_params.shift=-8;

	  arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,96,weight_11,bias_11,section,&section[258048]);

	  memcpy(section,&section[258048],24576);

	  fc_params.activation.max=127;
	  fc_params.activation.min=-128;
	  fc_params.input_offset=81;
	  fc_params.output_offset=-3;

	  t_quant_params.multiplier=1294720384;

	  arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_12,&bias_dims,bias_12,&output_dims,&section[282624]);

	  arm_nn_transpose_bnc_to_nbc_q7(256,6,48,&section[282624],section);

	  memcpy(&section[282624],section,0);

	  arm_nn_batch_mat_mult_nt_t_s8(&section[282624],&section[282624],NULL,section,1911858614,-6,256,48,256,3,3,-2,2,-128,127);

	  arm_softmax_s8(section,512,256,2044218880,-4,0,section);

	  arm_nn_batch_mat_mult_s8(&ctx,section,&section[282624],NULL,&section[73728],1118498490,-6,256,256,48,128,3,-1,2,-128,127);

	  arm_nn_transpose_bnc_to_nbc_q7(2,12288,1,&section[73728],&section[258048]);

	  fc_params.input_offset=1;
	  fc_params.output_offset=-20;

	  t_quant_params.multiplier=2040644608;
	  t_quant_params.shift=-9;

	  arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,&section[258048],&filter_dims,weight_13,&bias_dims,bias_13,&output_dims,section);

	  arm_elementwise_add_s8(section,&section[282624],20,1739629056,-6,128,1637537664,-5,0,&section[24576],-85,1891474688,-5,-128,127,24576);

	  memcpy(section,&section[24576],24576);

	  memcpy(&section[282624],section,24576);

	  norm_params.input_offset=85;
	  norm_params.output_offset=-34;

	  t_quant_params.multiplier=2111477760;

	  arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,96,weight_14,bias_14,section,&section[249856]);

	  fc_params.input_offset=34;
	  fc_params.output_offset=-128;

	  t_quant_params.multiplier=1742290816;
	  t_quant_params.shift=-7;

	  arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,&section[249856],&filter_dims,weight_15,&bias_dims,bias_15,&output_dims,section);

	  fc_params.input_offset=128;
	  fc_params.output_offset=0;

	  t_quant_params.multiplier=1919606272;
	  t_quant_params.shift=-9;

	  arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_16,&bias_dims,bias_16,&output_dims,&section[258048]);

	  arm_elementwise_add_s8(&section[258048],&section[282624],0,1175068032,-5,34,1261980032,-4,0,section,-29,1366739584,-4,-128,127,24576);

	  input_dims.c=96;

	  filter_dims.h=3;
	  filter_dims.w=3;
	  filter_dims.n=96;


	  conv_params.padding.h=1;
	  conv_params.padding.w=1;
	  conv_params.input_offset=29;

	  memcpy(conv_mult_use,mult_17,384);

	  memcpy(conv_shift_use,shift_17,384);

	  arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_17,&bias_dims,bias_17,&output_dims,&section[274432]);


	  filter_dims.h=1;
	  filter_dims.w=1;

	  output_dims.c=128;

	  conv_params.padding.h=0;
	  conv_params.padding.w=0;
	  conv_params.input_offset=128;

	  memcpy(conv_mult_use,mult_18,512);

	  memcpy(conv_shift_use,shift_18,512);

	  arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,&section[274432],&filter_dims,weight_18,&bias_dims,bias_18,&output_dims,section);

	  input_dims.c=128;

	  filter_dims.h=3;
	  filter_dims.w=3;


	  conv_params.padding.h=1;
	  conv_params.padding.w=1;

	  memcpy(conv_mult_use,mult_19,512);

	  memcpy(conv_shift_use,shift_19,512);

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[282624],&filter_dims,weight_19,&bias_dims,bias_19,&output_dims,section,117158);


	  filter_dims.h=1;
	  filter_dims.w=1;

	  output_dims.c=96;

	  conv_params.padding.h=0;
	  conv_params.padding.w=0;

	  memcpy(conv_mult_use,mult_20,384);

	  memcpy(conv_shift_use,shift_20,384);

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_20,&bias_dims,bias_20,&output_dims,&section[282624],9780);

	  memcpy(&section[282624],section,24576);

	  norm_params.input_offset=128;
	  norm_params.output_offset=-87;

	  t_quant_params.multiplier=2146633600;

	  arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,96,weight_21,bias_21,section,&section[258048]);

	  memcpy(section,&section[258048],24576);

	  fc_params.input_offset=87;
	  fc_params.output_offset=-5;

	  t_quant_params.multiplier=1438111872;
	  t_quant_params.shift=-8;

	  arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_22,&bias_dims,bias_22,&output_dims,&section[282624],21978);

	  arm_nn_transpose_bnc_to_nbc_q7(256,6,48,&section[282624],section);

	  memcpy(&section[282624],section,0);

	  arm_nn_batch_mat_mult_nt_t_s8(&section[282624],&section[282624],NULL,section,1532397482,-5,256,48,256,5,5,-7,2,-128,127);

	  arm_softmax_s8(section,512,256,1801219200,-4,0,section);

	  arm_nn_batch_mat_mult_s8(&ctx,section,&section[282624],NULL,&section[73728],1347038839,-6,256,256,48,128,5,-13,2,-128,127);

	  arm_nn_transpose_bnc_to_nbc_q7(2,12288,1,&section[73728],&section[258048]);

	  fc_params.input_offset=13;
	  fc_params.output_offset=-17;

	  t_quant_params.multiplier=2105830784;
	  t_quant_params.shift=-9;

	  arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[258048],&filter_dims,weight_23,&bias_dims,bias_23,&output_dims,section,7332);

	  arm_elementwise_add_s8(section,&section[282624],17,1114091520,-5,128,1598667008,-5,0,&section[24576],-75,1119693312,-4,-128,127,24576);

	  memcpy(section,&section[24576],24576);

	  memcpy(&section[282624],section,24576);

	  norm_params.input_offset=75;
	  norm_params.output_offset=-40;

	  t_quant_params.multiplier=1899635328;

	  arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,96,weight_24,bias_24,section,&section[266240]);

	  fc_params.input_offset=40;
	  fc_params.output_offset=-128;

	  t_quant_params.multiplier=1996709632;
	  t_quant_params.shift=-7;

	  arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[266240],&filter_dims,weight_25,&bias_dims,bias_25,&output_dims,section,9716);

	  fc_params.input_offset=128;
	  fc_params.output_offset=4;

	  t_quant_params.multiplier=1943961600;
	  t_quant_params.shift=-8;

	  arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_26,&bias_dims,bias_26,&output_dims,&section[249856],9706);

	  arm_elementwise_add_s8(&section[249856],&section[282624],-4,1656687232,-5,40,1399004160,-4,0,section,-32,1643182848,-4,-128,127,24576);

	  input_dims.c=96;

	  filter_dims.h=3;
	  filter_dims.w=3;


	  conv_params.padding.h=1;
	  conv_params.padding.w=1;
	  conv_params.input_offset=32;

	  memcpy(conv_mult_use,mult_27,384);

	  memcpy(conv_shift_use,shift_27,384);

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_27,&bias_dims,bias_27,&output_dims,&section[290816],65900);


	  filter_dims.h=1;
	  filter_dims.w=1;

	  output_dims.c=256;

	  conv_params.padding.h=0;
	  conv_params.padding.w=0;
	  conv_params.input_offset=128;

	  memcpy(conv_mult_use,mult_28,1024);

	  memcpy(conv_shift_use,shift_28,1024);

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[290816],&filter_dims,weight_28,&bias_dims,bias_28,&output_dims,section,19552);

	  input_dims.c=256;


	  output_dims.c=512;


	  memcpy(conv_mult_use,mult_29,2048);

	  memcpy(conv_shift_use,shift_29,2048);

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_29,&bias_dims,bias_29,&output_dims,&section[299008],104138);

	  memcpy(section,&section[299008],8192);


	  filter_dims.h=16;
	  filter_dims.w=16;

	  output_dims.h=1;
	  output_dims.w=1;

	  arm_avgpool_s8_with_quantization(&ctx,&pool_params,&input_dims,section,&filter_dims,&output_dims,128,-128,1105721945,3,&section[305664]);

	  memcpy(section,&section[305664],1536);

	  fc_params.output_offset=-19;

	  t_quant_params.multiplier=1330214144;
	  t_quant_params.shift=-9;

	  arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_30,&bias_dims,bias_30,&output_dims,&section[301056]);

	  memcpy(section,&section[301056],6144);

	  return 0;}


