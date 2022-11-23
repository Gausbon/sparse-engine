#include "func.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "stdio.h"

int32_t arm_convolve_s8_sparse_get_buffer_size (const cmsis_nn_dims *output_dims)
{
    return sizeof(q31_t) * (output_dims->w) * (output_dims->h);
}

void arm_nn_convolve_s8_single_sparse( const cmsis_nn_conv_params *conv_params,
                                    const int32_t stride_x_size,
                                    const int32_t stride_y_size,
                                    const q7_t *input_data,
                                    const int32_t input_x,
                                    const int32_t input_y,
                                    const int32_t output_x,
                                    const int32_t output_y,
                                    const int32_t input_ch,
                                    const int32_t in_ch,
                                    const int32_t h,
                                    const int32_t w,
                                    const int32_t ker_val,
                                    q31_t *buffer)
{
    const int32_t stride_x = conv_params->stride.w;
    const int32_t stride_y = conv_params->stride.h;
    const int32_t pad_x = conv_params->padding.w;
    const int32_t pad_y = conv_params->padding.h;
    const int32_t dilation_x = conv_params->dilation.w;
    const int32_t dilation_y = conv_params->dilation.h;
    const int32_t in_offset = conv_params->input_offset;

    int32_t k_y, k_x;
    int32_t k_y_size, k_x_size;
    q7_t input_val;

    k_y = dilation_y * h - pad_y;
    k_y_size = k_y * input_x * input_ch;
    
    for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++) {
        k_x = dilation_x * w - pad_x;
        k_x_size = k_x * input_ch;
        for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++) {
            if (k_y >= 0 && k_y < input_y && k_x >= 0 && k_x < input_x) {
                // not padding
                input_val = input_data[k_y_size + k_x_size + in_ch];
                (*buffer) += ((input_val + in_offset) * ker_val);
            }
            buffer++;
            k_x += stride_x;
            k_x_size += stride_x_size;
        }
        k_y += stride_y;
        k_y_size += stride_y_size;
    }
}


void arm_nn_convolve_s8_double_sparse( const cmsis_nn_conv_params *conv_params,
                                    const int32_t stride_x_size,
                                    const int32_t stride_y_size,
                                    const q7_t *input_data,
                                    const int32_t input_x,
                                    const int32_t input_y,
                                    const int32_t output_x,
                                    const int32_t output_y,
                                    const int32_t input_ch,
                                    const int32_t in_ch_0,
                                    const int32_t h_0,
                                    const int32_t w_0,
                                    const int32_t ker_val_0,
                                    const int32_t in_ch_1,
                                    const int32_t h_1,
                                    const int32_t w_1,
                                    const int32_t ker_val_1,
                                    q31_t *buffer)
{
    const int32_t stride_x = conv_params->stride.w;
    const int32_t stride_y = conv_params->stride.h;
    const int32_t pad_x = conv_params->padding.w;
    const int32_t pad_y = conv_params->padding.h;
    const int32_t dilation_x = conv_params->dilation.w;
    const int32_t dilation_y = conv_params->dilation.h;
    const int32_t in_offset = conv_params->input_offset;

    int32_t k_y_0, k_y_1, k_x_0, k_x_1;
    int32_t k_y_0_size, k_y_1_size, k_x_0_size, k_x_1_size;
    q7_t input_0, input_1;

    q31_t kernel_15x2 = __PKHBT(ker_val_0, ker_val_1, 16);
    q31_t offset_15x2 = __PKHBT(in_offset, in_offset, 16);

    k_y_0 = dilation_y * h_0 - pad_y;
    k_y_1 = dilation_y * h_1 - pad_y;
    k_y_0_size = k_y_0 * input_x * input_ch;
    k_y_1_size = k_y_1 * input_x * input_ch;

    for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++) {
        k_x_0 = dilation_x * w_0 - pad_x;
        k_x_1 = dilation_x * w_1 - pad_x;
        k_x_0_size = k_x_0 * input_ch;
        k_x_1_size = k_x_1 * input_ch;
        for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++) {
            if (k_y_0 >= 0 && k_y_0 < input_y && k_x_0 >= 0 && k_x_0 < input_x) {
                input_0 = input_data[k_y_0_size + k_x_0_size + in_ch_0];
            } else {
                // padding does not include offset
                input_0 = -in_offset;
            }

            if (k_y_1 >= 0 && k_y_1 < input_y && k_x_1 >= 0 && k_x_1 < input_x) {
                input_1 = input_data[k_y_1_size + k_x_1_size + in_ch_1];
            } else {
                input_1 = -in_offset;
            }

            if (!(input_0 == -in_offset && input_1 == -in_offset)) {
                int32_t input_15x2 = __PKHBT(input_0, input_1, 16);
                input_15x2 = __QADD16(input_15x2, offset_15x2);
                (*buffer) = __SMLAD(kernel_15x2, input_15x2, (*buffer));
            }
            buffer++;
            k_x_0 += stride_x;
            k_x_1 += stride_x;
            k_x_0_size += stride_x_size;
            k_x_1_size += stride_x_size;
        }
        k_y_0 += stride_y;
        k_y_1 += stride_y;
        k_y_0_size += stride_y_size;
        k_y_1_size += stride_y_size;
    }
}


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
                           const q31_t input_count,
                           const int32_t block)
{
    (void)bias_dims;
	  
    if (ctx->size < arm_convolve_s8_sparse_get_buffer_size(output_dims)) {
        return ARM_MATH_ARGUMENT_ERROR;
    }
    
    // assert block == 2 or block == 4
    // conv information
    
    const int32_t batch = input_dims->n;
    const int32_t input_ch = input_dims->c;
    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t output_ch = output_dims->c;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    
    const int32_t out_offset = conv_params->output_offset;
    const int32_t stride_x = conv_params->stride.w;
    const int32_t stride_y = conv_params->stride.h;
    const int32_t act_min = conv_params->activation.min;
    const int32_t act_max = conv_params->activation.max;
    
    q31_t *buffer = (q31_t *)(ctx->buf);
    q31_t *dec_buf = buffer + (arm_convolve_s8_sparse_get_buffer_size(output_dims) >> 2);
    q31_t *dec_end_ptr, *dec_ptr = dec_buf;
    q31_t *dec_buf_end = buffer + ((ctx->size) >> 2);

    q7_t last_val = 0, cur_val = 0;
    int32_t last_pos[4] = {0};
    int32_t cur_pos[4] = {0};

    q31_t *mult_ptr = quant_params->multiplier;
    q31_t *shift_ptr = quant_params->shift;

    int32_t output_size = output_x * output_y;
    int32_t stride_y_size = stride_y * input_x * input_ch;
    int32_t stride_x_size = stride_x * input_ch;
    
    int32_t block_cnt = 0;
    int32_t res = 0;

    for (int32_t i_batch = 0; i_batch < batch; ++i_batch) {
        memset(buffer, 0, sizeof(q31_t) * output_size);
        
        const q7_t *filter_ptr = filter_data;
        const q7_t *end_ptr = filter_ptr + input_count;
        const q7_t *in_ptr = &input_data[i_batch * input_x * input_y * input_ch];
        q7_t *out_ptr = &output_data[i_batch * output_x * output_y * output_ch];

        block_cnt = 0;
        
        while (1) {
            // decode procedure
// decode procedure
            if (dec_ptr < dec_end_ptr) {
                if (block_cnt == 0) {
                    memcpy(last_pos, dec_ptr, 16);
                    dec_ptr += 4;
                    last_val = (q7_t) (*dec_ptr++);
                    if (cur_pos[3] != last_pos[3]) {
                        arm_nn_output_per_channel ( cur_pos[3], last_pos[3], out_offset, output_size,
                            output_ch, act_min, act_max, bias_data, mult_ptr, shift_ptr, buffer,
                            out_ptr);
                    }
                    memcpy(cur_pos, last_pos, 16);
                    ++cur_pos[0];
                    cur_val = (q7_t) (*dec_ptr++);

                    block_cnt = (block_cnt + 2) % block;
                } else {
                    last_pos[0] += 2;
                    last_val = (q7_t) (*dec_ptr++);

                    cur_pos[0] += 2;
                    cur_val = (q7_t) (*dec_ptr++);

                    block_cnt = (block_cnt + 2) % block;
                }
            } else {
                dec_ptr = dec_buf;
                //printf("start\r\n");
                dec_end_ptr = arm_nn_decode_4d (dec_buf, dec_buf_end,
                        &filter_ptr, end_ptr,
                        cur_pos, &res,
                        input_ch, kernel_x,
                        kernel_y, block);
                if (dec_end_ptr == dec_buf) {
                    break;
                }
                continue;
            }
            

            arm_nn_convolve_s8_double_sparse(conv_params,
                    stride_x_size, stride_y_size,
                    in_ptr,
                    input_x, input_y,
                    output_x, output_y,
                    input_ch,
                    last_pos[0], last_pos[1],
                    last_pos[2], last_val,
                    cur_pos[0], cur_pos[1],
                    cur_pos[2], cur_val,
                    buffer);
        }

        // start to output finally  
        arm_nn_output_per_channel ( cur_pos[3], output_ch, out_offset, output_size,
                    output_ch, act_min, act_max, bias_data, mult_ptr, shift_ptr, buffer,
                    out_ptr);

        buffer = (q31_t *)(ctx->buf);
        memset(buffer, 0, sizeof(q31_t) * output_size);
    }

    return ARM_MATH_SUCCESS;
}
