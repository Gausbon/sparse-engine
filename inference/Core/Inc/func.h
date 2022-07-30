#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#ifndef __FUNC_H
#define __FUNC_H

// quant inference
int quantization_inference(void);

// cmsis nn func
typedef struct
{
    int32_t input_offset;
    int32_t output_offset;
    cmsis_nn_activation activation;
} cmsis_nn_layernorm_params;

arm_cmsis_nn_status arm_nn_batch_mat_mult_nt_t_s8(const q7_t *lhs,
                                            const q7_t *rhs,
                                            const q31_t *bias,
                                            q7_t *dst,
                                            const int32_t dst_multiplier,
                                            const int32_t dst_shift,
                                            const int32_t lhs_rows,
                                            const int32_t lhs_cols,
                                            const int32_t rhs_rows,
                                            const int32_t lhs_offset,
                                            const int32_t rhs_offset,
                                            const int32_t dst_offset,
                                            const int32_t batch,
                                            const int32_t activation_min,
                                            const int32_t activation_max);


arm_cmsis_nn_status arm_nn_batch_mat_mult_s8(const cmsis_nn_context *ctx,
                                            const q7_t *lhs,
                                            const q7_t *rhs,
                                            const q31_t *bias,
                                            q7_t *dst,
                                            const int32_t dst_multiplier,
                                            const int32_t dst_shift,
                                            const int32_t lhs_rows,
                                            const int32_t lhs_cols,
                                            const int32_t rhs_cols,
                                            const int32_t lhs_offset,
                                            const int32_t rhs_offset,
                                            const int32_t dst_offset,
                                            const int32_t batch,
                                            const int32_t activation_min,
                                            const int32_t activation_max);


arm_status arm_nn_layernorm_s8 (const cmsis_nn_context *ctx,
                           const cmsis_nn_layernorm_params *layernorm_params,
                           const cmsis_nn_per_tensor_quant_params *quant_params,
                           const int32_t dim_b,
                           const int32_t dim_c,
                           const q7_t *weight,
                           const q31_t *bias,
                           const q7_t *input_data,
                           q7_t *output_data);


void arm_nn_output_per_channel (   const int32_t start_channel,
                                const int32_t end_channel,
                                const int32_t out_offset,
                                const int32_t output_count,
                                const int32_t output_ch,
                                const int32_t act_min,
                                const int32_t act_max,
                                const int32_t *bias_data,
                                const int32_t *mult_data,
                                const int32_t *shift_data,
                                const q31_t *input_data,
                                q7_t *output_data);


void arm_nn_sparse_decode_4d(    const int32_t last_in_ch,
                                    const int32_t last_h,
                                    const int32_t last_w,
                                    const int32_t last_out_ch,
                                    const int32_t input_ch,
                                    const int32_t kernel_x,
                                    const int32_t kernel_y,
                                    const q7_t **filter_data,
                                    int32_t *cur_in_ch,
                                    int32_t *cur_h,
                                    int32_t *cur_w,
                                    int32_t *cur_out_ch,
                                    int32_t *mat_flag,
                                    int32_t *counter,
                                    q7_t *cur_val);


void arm_nn_sparse_decode_2d(    const int32_t last_in_ch,
                                    const int32_t last_out_ch,
                                    const int32_t input_ch,
                                    const q7_t **filter_data,
                                    int32_t *cur_in_ch,
                                    int32_t *cur_out_ch,
                                    int32_t *mat_flag,
                                    int32_t *counter,
                                    q7_t *cur_val);


arm_cmsis_nn_status arm_elementwise_add_s8_with_neg(const int8_t *input_1_vect,
                                           const int8_t *input_2_vect,
                                           const int32_t input_1_offset,
                                           const int32_t input_1_mult,
                                           const int32_t input_1_shift,
                                           const int32_t input_2_offset,
                                           const int32_t input_2_mult,
                                           const int32_t input_2_shift,
                                           const int32_t left_shift,
                                           int8_t *output,
                                           const int32_t out_offset,
                                           const int32_t out_mult,
                                           const int32_t out_shift,
                                           const int32_t out_activation_min,
                                           const int32_t out_activation_max,
                                           const int32_t block_size);


arm_cmsis_nn_status arm_nn_transpose_bnc_to_nbc_q7(const int32_t dim_b, 
                                            const int32_t dim_n, 
                                            const int32_t dim_c, 
                                            const q7_t *input_section, 
                                            q7_t *output_section);


// cmsis func
int32_t arm_convolve_s8_sparse_get_buffer_size (const cmsis_nn_dims *output_dims);


arm_status arm_convolve_s8_sparse (const cmsis_nn_context *ctx,
                           const cmsis_nn_conv_params *conv_params,
                           const cmsis_nn_per_channel_quant_params *quant_params,
                           const cmsis_nn_dims *input_dims,
                           const q7_t *input_data,
                           const cmsis_nn_dims *filter_dims,
                           const q7_t *filter_data,
                           const cmsis_nn_dims *bias_dims,
                           const int32_t *bias_data,
                           const cmsis_nn_dims *output_dims,
                           q7_t *output_data,
                           const q31_t input_count);

                          
arm_status arm_depthwise_conv_s8_sparse (const cmsis_nn_context *ctx,
                           const cmsis_nn_dw_conv_params *dw_conv_params, 
                           const cmsis_nn_per_channel_quant_params *quant_params,
                           const cmsis_nn_dims *input_dims,
                           const q7_t *input_data,
                           const cmsis_nn_dims *filter_dims,
                           const q7_t *filter_data,
                           const cmsis_nn_dims *bias_dims,
                           const int32_t *bias_data,
                           const cmsis_nn_dims *output_dims,
                           q7_t *output_data,
                           const q31_t input_count);

int32_t arm_fc_s8_sparse_get_buffer_size(const cmsis_nn_dims *output_dims);


arm_status arm_fully_connected_s8_sparse (const cmsis_nn_context *ctx,
                            const cmsis_nn_fc_params *fc_params,
                            const cmsis_nn_per_tensor_quant_params *quant_params,
                            const cmsis_nn_dims *input_dims,
                            const q7_t *input_data,
                            const cmsis_nn_dims *filter_dims,
                            const q7_t *filter_data,
                            const cmsis_nn_dims *bias_dims,
                            const int32_t *bias_data,
                            const cmsis_nn_dims *output_dims,
                            q7_t *output_data,
                            const int32_t input_count);


arm_status arm_avgpool_s8_with_quantization (const cmsis_nn_context *ctx,
                          const cmsis_nn_pool_params *pool_params,
                          const cmsis_nn_dims *input_dims,
                          const q7_t *src,
                          const cmsis_nn_dims *filter_dims,
                          const cmsis_nn_dims *output_dims,
                          const int32_t input_offset,
                          const int32_t output_offset,
                          const int32_t multiplier,
                          const int32_t shift,
                          q7_t *dst);


arm_status arm_maxpool_s8_with_quantization (const cmsis_nn_context *ctx,
                          const cmsis_nn_pool_params *pool_params,
                          const cmsis_nn_dims *input_dims,
                          const q7_t *src,
                          const cmsis_nn_dims *filter_dims,
                          const cmsis_nn_dims *output_dims,
                          const int32_t input_offset,
                          const int32_t output_offset,
                          const int32_t multiplier,
                          const int32_t shift,
                          q7_t *dst);


int result_check(const cmsis_nn_context *ctx,
                          const q7_t *section,
                          const int32_t class);
                          

#endif
