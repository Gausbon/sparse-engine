#include "arm_nnfunctions.h"
#include "data.h"
#include "func.h"

int quantization_inference(void) {
    q7_t section[332*1024] = {0};
    q7_t buf[4*32*32] = {0};
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

	  c_quant_params.multiplier=mult_0;
	  c_quant_params.shift=shift_0;

	  arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_0,&bias_dims,bias_0,&output_dims,&section[339958]);

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

	  c_quant_params.multiplier=mult_1;
	  c_quant_params.shift=shift_1;

	  arm_depthwise_conv_s8(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[339958],&filter_dims,weight_1,&bias_dims,bias_1,&output_dims,section);

	  input_dims.h=16;
	  input_dims.w=16;
	  input_dims.c=6;

	  filter_dims.h=1;
	  filter_dims.w=1;

	  output_dims.c=64;

	  conv_params.input_offset=128;
	  conv_params.output_offset=-12;

	  c_quant_params.multiplier=mult_2;
	  c_quant_params.shift=shift_2;

	  arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_2,&bias_dims,bias_2,&output_dims,&section[339712]);

	  memcpy(section,&section[339712],256);

	  input_dims.c=64;


	  output_dims.c=192;

	  conv_params.input_offset=12;
	  conv_params.output_offset=-128;

	  c_quant_params.multiplier=mult_3;
	  c_quant_params.shift=shift_3;

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_3,&bias_dims,bias_3,&output_dims,&section[339968],12288);

	  input_dims.c=1;

	  filter_dims.h=3;
	  filter_dims.w=3;


	  dw_conv_params.stride.h=1;
	  dw_conv_params.stride.w=1;

	  c_quant_params.multiplier=mult_4;
	  c_quant_params.shift=shift_4;

	  arm_depthwise_conv_s8_sparse(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[339968],&filter_dims,weight_4,&bias_dims,bias_4,&output_dims,section,1728);

	  input_dims.c=192;

	  filter_dims.h=1;
	  filter_dims.w=1;

	  output_dims.c=256;

	  conv_params.input_offset=128;
	  conv_params.output_offset=-2;

	  c_quant_params.multiplier=mult_5;
	  c_quant_params.shift=shift_5;

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_5,&bias_dims,bias_5,&output_dims,&section[208896],49152);

	  memcpy(section,&section[208896],131072);

	  input_dims.c=256;


	  output_dims.c=512;

	  conv_params.input_offset=2;
	  conv_params.output_offset=-128;

	  c_quant_params.multiplier=mult_6;
	  c_quant_params.shift=shift_6;

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_6,&bias_dims,bias_6,&output_dims,&section[208896],131072);

	  input_dims.c=1;

	  filter_dims.h=3;
	  filter_dims.w=3;



	  c_quant_params.multiplier=mult_7;
	  c_quant_params.shift=shift_7;

	  arm_depthwise_conv_s8_sparse(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[208896],&filter_dims,weight_7,&bias_dims,bias_7,&output_dims,section,4608);

	  input_dims.c=512;

	  filter_dims.h=1;
	  filter_dims.w=1;

	  output_dims.c=32;

	  conv_params.input_offset=128;
	  conv_params.output_offset=0;

	  c_quant_params.multiplier=mult_8;
	  c_quant_params.shift=shift_8;

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_8,&bias_dims,bias_8,&output_dims,&section[315392],16384);

	  memcpy(section,&section[315392],24576);

	  input_dims.c=32;

	  filter_dims.h=3;
	  filter_dims.w=3;


	  conv_params.padding.h=1;
	  conv_params.padding.w=1;
	  conv_params.input_offset=0;
	  conv_params.output_offset=-128;

	  c_quant_params.multiplier=mult_9;
	  c_quant_params.shift=shift_9;

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[315392],&filter_dims,weight_9,&bias_dims,bias_9,&output_dims,section,9216);


	  filter_dims.h=1;
	  filter_dims.w=1;

	  output_dims.c=64;

	  conv_params.padding.h=0;
	  conv_params.padding.w=0;
	  conv_params.input_offset=128;

	  c_quant_params.multiplier=mult_10;
	  c_quant_params.shift=shift_10;

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_10,&bias_dims,bias_10,&output_dims,&section[315392],2048);

	  memcpy(&section[315392],section,24576);

	  norm_params.activation.max=127;
	  norm_params.activation.min=-128;
	  norm_params.input_offset=128;
	  norm_params.output_offset=-74;

	  t_quant_params.multiplier=1376040192;
	  t_quant_params.shift=-8;

	  arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,64,weight_11,bias_11,section,&section[282624]);

	  memcpy(section,&section[282624],32768);

	  fc_params.activation.max=127;
	  fc_params.activation.min=-128;
	  fc_params.input_offset=74;
	  fc_params.output_offset=-5;

	  t_quant_params.multiplier=1488163072;

	  arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_12,&bias_dims,bias_12,&output_dims,&section[315392],12288);

	  arm_nn_transpose_bnc_to_nbc_q7(256,12,16,&section[315392],section);

	  memcpy(&section[315392],section,0);

	  arm_nn_batch_mat_mult_nt_t_s8(&section[315392],&section[315392],NULL,section,1382741133,-5,256,16,256,5,5,-1,4,-128,127);

	  arm_softmax_s8(section,1024,256,2001747328,-4,0,section);

	  arm_nn_batch_mat_mult_s8(&ctx,section,&section[315392],NULL,&section[24576],1400183945,-6,256,256,16,128,5,-4,4,-128,127);

	  arm_nn_transpose_bnc_to_nbc_q7(4,4096,1,&section[24576],&section[184320]);

	  fc_params.input_offset=4;
	  fc_params.output_offset=-16;

	  t_quant_params.multiplier=1576938496;

	  arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[184320],&filter_dims,weight_13,&bias_dims,bias_13,&output_dims,section,4096);

	  arm_elementwise_add_s8(section,&section[315392],16,1425808384,-6,128,1636769920,-5,0,&section[24576],-89,1789672960,-5,-128,127,24576);

	  memcpy(section,&section[24576],24576);

	  memcpy(&section[315392],section,24576);

	  norm_params.input_offset=89;
	  norm_params.output_offset=-45;

	  t_quant_params.multiplier=1223694208;

	  arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,64,weight_14,bias_14,section,&section[241664]);

	  fc_params.input_offset=45;
	  fc_params.output_offset=-128;

	  t_quant_params.multiplier=1383175552;
	  t_quant_params.shift=-6;

	  arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[241664],&filter_dims,weight_15,&bias_dims,bias_15,&output_dims,section,32768);

	  fc_params.input_offset=128;
	  fc_params.output_offset=-2;

	  t_quant_params.multiplier=1173848064;
	  t_quant_params.shift=-9;

	  arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_16,&bias_dims,bias_16,&output_dims,&section[290816],32768);

	  arm_elementwise_add_s8(&section[290816],&section[315392],2,1942912640,-5,45,1177611264,-4,0,section,-25,1486402048,-4,-128,127,24576);

	  input_dims.c=64;

	  filter_dims.h=3;
	  filter_dims.w=3;
	  filter_dims.n=64;


	  conv_params.padding.h=1;
	  conv_params.padding.w=1;
	  conv_params.input_offset=25;

	  c_quant_params.multiplier=mult_17;
	  c_quant_params.shift=shift_17;

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_17,&bias_dims,bias_17,&output_dims,&section[307200],36864);


	  filter_dims.h=1;
	  filter_dims.w=1;

	  output_dims.c=128;

	  conv_params.padding.h=0;
	  conv_params.padding.w=0;
	  conv_params.input_offset=128;

	  c_quant_params.multiplier=mult_18;
	  c_quant_params.shift=shift_18;

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[307200],&filter_dims,weight_18,&bias_dims,bias_18,&output_dims,section,8192);

	  input_dims.c=128;

	  filter_dims.h=3;
	  filter_dims.w=3;


	  conv_params.padding.h=1;
	  conv_params.padding.w=1;

	  c_quant_params.multiplier=mult_19;
	  c_quant_params.shift=shift_19;

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[323584],&filter_dims,weight_19,&bias_dims,bias_19,&output_dims,section,147456);


	  filter_dims.h=1;
	  filter_dims.w=1;

	  output_dims.c=96;

	  conv_params.padding.h=0;
	  conv_params.padding.w=0;

	  c_quant_params.multiplier=mult_20;
	  c_quant_params.shift=shift_20;

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_20,&bias_dims,bias_20,&output_dims,&section[323584],12288);

	  memcpy(&section[323584],section,16384);

	  norm_params.input_offset=128;
	  norm_params.output_offset=-93;

	  t_quant_params.multiplier=1123438976;
	  t_quant_params.shift=-8;

	  arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,96,weight_21,bias_21,section,&section[192512]);

	  memcpy(section,&section[192512],131072);

	  fc_params.input_offset=93;
	  fc_params.output_offset=-5;

	  t_quant_params.multiplier=1480947584;

	  arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_22,&bias_dims,bias_22,&output_dims,&section[323584],27648);

	  arm_nn_transpose_bnc_to_nbc_q7(256,6,48,&section[323584],section);

	  memcpy(&section[323584],section,0);

	  arm_nn_batch_mat_mult_nt_t_s8(&section[323584],&section[323584],NULL,section,1912130266,-6,256,48,256,5,5,-4,2,-128,127);

	  arm_softmax_s8(section,512,256,1737567872,-4,0,section);

	  arm_nn_batch_mat_mult_s8(&ctx,section,&section[323584],NULL,&section[16384],1359316022,-6,256,256,48,128,5,-2,2,-128,127);

	  arm_nn_transpose_bnc_to_nbc_q7(2,12288,1,&section[16384],&section[61440]);

	  fc_params.input_offset=2;
	  fc_params.output_offset=3;

	  t_quant_params.multiplier=1853947776;
	  t_quant_params.shift=-9;

	  arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[61440],&filter_dims,weight_23,&bias_dims,bias_23,&output_dims,section,9216);

	  arm_elementwise_add_s8(section,&section[323584],-3,1545222272,-6,128,1295821824,-4,0,&section[16384],-95,1444939904,-4,-128,127,16384);

	  memcpy(section,&section[16384],16384);

	  memcpy(&section[323584],section,16384);

	  norm_params.input_offset=95;
	  norm_params.output_offset=-59;

	  t_quant_params.multiplier=1918940544;

	  arm_nn_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,96,weight_24,bias_24,section,&section[274432]);

	  fc_params.input_offset=59;
	  fc_params.output_offset=-128;

	  t_quant_params.multiplier=1127827712;
	  t_quant_params.shift=-6;

	  arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,&section[274432],&filter_dims,weight_25,&bias_dims,bias_25,&output_dims,section,12288);

	  fc_params.input_offset=128;
	  fc_params.output_offset=-14;

	  t_quant_params.multiplier=1542181760;
	  t_quant_params.shift=-8;

	  arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_26,&bias_dims,bias_26,&output_dims,&section[307200],12288);

	  arm_elementwise_add_s8(&section[307200],&section[323584],14,1364048640,-5,59,1415619200,-4,0,section,-56,1668544896,-4,-128,127,16384);

	  input_dims.c=96;

	  filter_dims.h=3;
	  filter_dims.w=3;
	  filter_dims.n=96;


	  conv_params.padding.h=1;
	  conv_params.padding.w=1;
	  conv_params.input_offset=56;

	  c_quant_params.multiplier=mult_27;
	  c_quant_params.shift=shift_27;

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_27,&bias_dims,bias_27,&output_dims,&section[331776],82944);


	  filter_dims.h=1;
	  filter_dims.w=1;

	  output_dims.c=128;

	  conv_params.padding.h=0;
	  conv_params.padding.w=0;
	  conv_params.input_offset=128;

	  c_quant_params.multiplier=mult_28;
	  c_quant_params.shift=shift_28;

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,&section[331776],&filter_dims,weight_28,&bias_dims,bias_28,&output_dims,section,12288);

	  input_dims.c=128;


	  output_dims.c=512;


	  c_quant_params.multiplier=mult_29;
	  c_quant_params.shift=shift_29;

	  arm_convolve_s8_sparse(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_29,&bias_dims,bias_29,&output_dims,&section[208896],65536);

	  memcpy(section,&section[208896],131072);

	  memcpy(&section[290816],section,49152);

	  fc_params.output_offset=108;

	  t_quant_params.multiplier=1904920448;
	  t_quant_params.shift=-11;

	  arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_30,&bias_dims,bias_30,&output_dims,&section[225280]);

	  arm_softmax_s8(&section[225280],1,256,2020102016,-3,0,section);

	  arm_nn_batch_mat_mult_s8(&ctx,section,&section[290816],NULL,&section[65536],2006289064,-7,1,256,512,128,128,-128,1,-128,127);

	  memcpy(section,&section[65536],65536);

	  fc_params.output_offset=-6;

	  t_quant_params.multiplier=1606091008;
	  t_quant_params.shift=-9;

	  arm_fully_connected_s8_sparse(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_31,&bias_dims,bias_31,&output_dims,&section[333824],5120);

	  memcpy(section,&section[333824],6144);

	  return 0;}


