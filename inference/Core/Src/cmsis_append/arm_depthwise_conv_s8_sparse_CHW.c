#include "func.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

# define DW_BLOCK 3
void arm_nn_dw_convolve_s8_single_sparse_1x1_CHW ( const cmsis_nn_dw_conv_params *dw_conv_params,
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
                                    const int32_t input_size,
                                    q31_t *buffer)
{
    const int32_t pad_x = dw_conv_params->padding.w;
    const int32_t pad_y = dw_conv_params->padding.h;
    const int32_t in_offset = dw_conv_params->input_offset;

    int32_t k_y, k_x;
    const q7_t *input_ptr, *input_start;

    input_start = input_data + in_ch * input_size;
    if ((k_y = h - pad_y) > 0) {
        input_start += (k_y * input_x);
    }
    if ((k_x = w - pad_x) > 0) {
        input_start += k_x;
    }

    for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++) {
        k_x = w - pad_x;
        input_ptr = input_start;
        if (k_y >= 0 && k_y < input_y) {
            for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++) {
                if (k_x >= 0 && k_x < input_x) {
                    // not padding
                    (*buffer) += ((*input_ptr++ + in_offset) * ker_val);
                }
                ++buffer;
                ++k_x;
            }
            input_start += input_x;
        } else {
            buffer += output_x;
        }
        ++k_y;
    }
}


void arm_nn_dw_convolve_s8_double_sparse_1x1_CHW ( const cmsis_nn_dw_conv_params *dw_conv_params,
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
                                    const int32_t input_size,
                                    q31_t *buffer)
{
    const int32_t pad_x = dw_conv_params->padding.w;
    const int32_t pad_y = dw_conv_params->padding.h;
    const int32_t in_offset = dw_conv_params->input_offset;

    int32_t k_y_0, k_x_0, k_y_1, k_x_1;
    q7_t input_0, input_1;
    const q7_t *input_start_0, *input_start_1;
    const q7_t *input_ptr_0, *input_ptr_1;
    q31_t kernel_15x2 = __PKHBT(ker_val_0, ker_val_1, 16);
    q31_t offset_15x2 = __PKHBT(in_offset, in_offset, 16);

    input_start_0 = input_data + in_ch_0 * input_size;
    input_start_1 = input_data + in_ch_1 * input_size;

    if ((k_y_0 = h_0 - pad_y) > 0) {
        input_start_0 += (k_y_0 * input_x);
    }
    if ((k_x_0 = w_0 - pad_x) > 0) {
        input_start_0 += k_x_0;
    }
    if ((k_y_1 = h_1 - pad_y) > 0) {
        input_start_1 += (k_y_1 * input_x);
    }
    if ((k_x_1 = w_1 - pad_x) > 0) {
        input_start_1 += k_x_1;
    }

    for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++) {
        k_x_0 = w_0 - pad_x;
        k_x_1 = w_1 - pad_x;
        input_ptr_0 = input_start_0;
        input_ptr_1 = input_start_1;
        // not both padding
        if ((k_y_0 >= 0 && k_y_0 < input_y) || (k_y_1 >= 0 && k_y_1 < input_y)) {
            for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++) {
                if (k_y_0 >= 0 && k_y_0 < input_y && k_x_0 >= 0 && k_x_0 < input_x) {
                    input_0 = *input_ptr_0++;
                } else {
                    input_0 = -in_offset;
                }

                if (k_y_1 >= 0 && k_y_1 < input_y && k_x_1 >= 0 && k_x_1 < input_x) {
                    input_1 = *input_ptr_1++;
                } else {
                    input_1 = -in_offset;
                }
                
                int32_t input_15x2 = __PKHBT(input_0, input_1, 16);
                input_15x2 = __QADD16(input_15x2, offset_15x2);
                (*buffer) = __SMLAD(kernel_15x2, input_15x2, (*buffer));
                
                ++buffer;
                ++k_x_0;
                ++k_x_1;
            }

            if (k_y_0 >= 0 && k_y_0 < input_y) {
                input_start_0 += input_x;
            }
            if (k_y_1 >= 0 && k_y_1 < input_y) {
                input_start_1 += input_x;
            }
        } else {
            buffer += input_x;
        }
        ++k_y_0;
        ++k_y_1;
    }
}


arm_status arm_depthwise_conv_s8_sparse_1x1_CHW (const cmsis_nn_context *ctx,
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
                           const q31_t input_count)
{
    (void)bias_dims;

    if (ctx->size < arm_convolve_s8_sparse_get_buffer_size(output_dims)) {
        return ARM_MATH_ARGUMENT_ERROR;
    }

    if (dw_conv_params->stride.w != 1 || dw_conv_params->stride.h != 1 || dw_conv_params->dilation.w != 1
        || dw_conv_params->dilation.h != 1) {
        return ARM_MATH_ARGUMENT_ERROR;
    }

    // conv information

    const int32_t batch = input_dims->n;
    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t output_ch = output_dims->c;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;

    const int32_t out_offset = dw_conv_params->output_offset;
    const int32_t act_min = dw_conv_params->activation.min;
    const int32_t act_max = dw_conv_params->activation.max;

    q31_t *buffer = (q31_t *)(ctx->buf);

    int32_t double_flag = 0;
    q7_t last_val = 0, cur_val = 0;
    int32_t last_out_ch = 0, cur_out_ch = 0;
    int32_t last_h = 0, cur_h = 0;
    int32_t last_w = 0, cur_w = 0;
    int32_t last_in_ch = 0, cur_in_ch = 0;
    int32_t mat_flag = 0;

    q31_t *mult_ptr = quant_params->multiplier;
    q31_t *shift_ptr = quant_params->shift;

    int32_t output_size = output_x * output_y;
    int32_t input_size = input_x * input_y;
    int32_t block_cnt = 0;

    for (int32_t i_batch = 0; i_batch < batch; ++i_batch) {
        memset(buffer, 0, sizeof(q31_t) * output_size);
        
        const q7_t *filter_ptr = filter_data;
        const q7_t *end_ptr = filter_ptr + input_count;
        const q7_t *in_ptr = &input_data[i_batch * input_size * output_ch];
        q7_t *out_ptr = &output_data[i_batch * output_size * output_ch];

        last_val = 0;
        last_out_ch = 0;
        last_in_ch = 0;
        last_h = 0;
        last_w = 0;

        double_flag = 0;
        mat_flag = 0;
        block_cnt = 0;
        
        while (filter_ptr < end_ptr) {
            // decode procedure
            cur_out_ch = last_out_ch;
            cur_h = last_h;
            cur_w = last_w;
            if (block_cnt == 0) {
                cur_in_ch = (*filter_ptr++) + last_in_ch + 128;
                cur_val = (*filter_ptr++);

                if (cur_val == 0) {
                    block_cnt = DW_BLOCK - 1;
                }
            } else {
                cur_in_ch = last_in_ch + 1;
                cur_val = (*filter_ptr++);
            }

            if (++block_cnt >= DW_BLOCK) {
                block_cnt = 0;
            }

            while (cur_in_ch >= 1) {
                cur_w += cur_in_ch;
                cur_in_ch = 0;
                while (cur_w >= kernel_x) {
                    ++cur_h;
                    cur_w -= kernel_x;
                    while (cur_h >= kernel_y) {
                        ++cur_out_ch;
                        cur_h -= kernel_y;
                        mat_flag = 1;
                    }
                }
            }

            if (mat_flag) {   
                // change the output channel, last output channel conv is done
                if (double_flag == 1 && last_val != 0) {
                    // remain one val to conv (last_val)
                    // in depthwise, out_ch replace in_ch
                    arm_nn_dw_convolve_s8_single_sparse_1x1_CHW(dw_conv_params,
                        in_ptr,
                        input_x, input_y,
                        output_x, output_y,
                        output_ch,
                        last_out_ch, last_h,
                        last_w, last_val,
                        input_size,
                        buffer);
                }
                double_flag = 0;

                // start to output
                arm_nn_output_per_channel_CHW ( last_out_ch, cur_out_ch, out_offset, output_size,
                            output_ch, act_min, act_max, bias_data, mult_ptr, shift_ptr, buffer,
                            out_ptr);

                // reset the buffer and flag
                buffer = (q31_t *)(ctx->buf);
                memset(buffer, 0, sizeof(q31_t) * output_size);
                mat_flag = 0;
            }

            if (double_flag && !(last_val == 0 && cur_val == 0)) {
                // use SIMD instructions to compute two val (last and cur val)
                // in depthwise, out_ch replace in_ch
                arm_nn_dw_convolve_s8_double_sparse_1x1_CHW(dw_conv_params,
                        in_ptr,
                        input_x, input_y,
                        output_x, output_y,
                        output_ch,
                        last_out_ch, last_h,
                        last_w, last_val,
                        cur_out_ch, cur_h,
                        cur_w, cur_val,
                        input_size,
                        buffer);
            }

            double_flag = 1 - double_flag;                
            last_out_ch = cur_out_ch;
            last_h = cur_h;
            last_w = cur_w;
            last_in_ch = cur_in_ch;
            last_val = cur_val;
        }

        if (double_flag) {
            // remain one val to conv (cur_val)
            // in depthwise, out_ch replace in_ch
            arm_nn_dw_convolve_s8_single_sparse_1x1_CHW(dw_conv_params,
                        in_ptr,
                        input_x, input_y,
                        output_x, output_y,
                        output_ch,
                        cur_out_ch, cur_h,
                        cur_w, cur_val,
                        input_size,
                        buffer);
        }

        double_flag = 0;

        // start to output finally  
        arm_nn_output_per_channel_CHW (cur_out_ch, output_ch, out_offset, output_size,
                    output_ch, act_min, act_max, bias_data, mult_ptr, shift_ptr, buffer,
                    out_ptr);

        buffer = (q31_t *)(ctx->buf);
        memset(buffer, 0, sizeof(q31_t) * output_size);
    }

    return ARM_MATH_SUCCESS;
}
