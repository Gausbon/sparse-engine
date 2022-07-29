#include "func.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
void arm_nn_dw_convolve_s8_single_sparse( const cmsis_nn_dw_conv_params *dw_conv_params,
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
    const int32_t stride_x = dw_conv_params->stride.w;
    const int32_t stride_y = dw_conv_params->stride.h;
    const int32_t pad_x = dw_conv_params->padding.w;
    const int32_t pad_y = dw_conv_params->padding.h;
    const int32_t dilation_x = dw_conv_params->dilation.w;
    const int32_t dilation_y = dw_conv_params->dilation.h;
    const int32_t in_offset = dw_conv_params->input_offset;

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
                (*buffer) = __QADD((__QADD(input_val, in_offset) * ker_val), (*buffer));
            }
            buffer++;
            k_x += stride_x;
            k_x_size += stride_x_size;
        }
        k_y += stride_y;
        k_y_size += stride_y_size;
    }
}


void arm_nn_dw_convolve_s8_double_sparse( const cmsis_nn_dw_conv_params *dw_conv_params,
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
    const int32_t stride_x = dw_conv_params->stride.w;
    const int32_t stride_y = dw_conv_params->stride.h;
    const int32_t pad_x = dw_conv_params->padding.w;
    const int32_t pad_y = dw_conv_params->padding.h;
    const int32_t dilation_x = dw_conv_params->dilation.w;
    const int32_t dilation_y = dw_conv_params->dilation.h;
    const int32_t in_offset = dw_conv_params->input_offset;

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
                           const q31_t input_count)
{
    (void)bias_dims;
	  
    if (ctx->size < arm_convolve_s8_sparse_get_buffer_size(output_dims)) {
        return ARM_MATH_ARGUMENT_ERROR;
    }
    
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
    
    const int32_t out_offset = dw_conv_params->output_offset;
    const int32_t stride_x = dw_conv_params->stride.w;
    const int32_t stride_y = dw_conv_params->stride.h;
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

    int32_t output_count = output_x * output_y;
    int32_t stride_y_size = stride_y * input_x * input_ch;
    int32_t stride_x_size = stride_x * input_ch;
    
    int32_t counter;

    for (int32_t i_batch = 0; i_batch < batch; ++i_batch) {
        memset(buffer, 0, sizeof(q31_t) * output_count);
        
        const q7_t *filter_ptr = &filter_data[0];
        const q7_t *in_ptr = &input_data[i_batch * input_x * input_y * input_ch];
        q7_t *out_ptr = &output_data[i_batch * output_x * output_y * output_ch];

        last_val = 0;
        last_out_ch = 0;
        last_in_ch = 0;
        last_h = 0;
        last_w = 0;

        double_flag = 0;
        mat_flag = 0;
        counter = input_count;

        while (counter) {
            // decode procedure
            arm_nn_sparse_decode_4d(
                last_in_ch, last_h,
                last_w, last_out_ch,
                input_ch, kernel_x,
                kernel_y, &filter_ptr,
                &cur_in_ch, &cur_h,
                &cur_w, &cur_out_ch,
                &mat_flag, &counter,
                &cur_val);

            if (mat_flag) {   
                // change the output channel, last output channel conv is done
                if (double_flag == 1 && last_val != 0) {
                    // remain one val to conv (last_val)
                    // in depthwise, out_ch replace in_ch
                    arm_nn_dw_convolve_s8_single_sparse(dw_conv_params,
                        stride_x_size, stride_y_size,
                        in_ptr,
                        input_x, input_y,
                        output_x, output_y,
                        input_ch,
                        last_out_ch, last_h,
                        last_w, last_val,
                        buffer);
                }
                double_flag = 0;

                // start to output
                arm_nn_output_per_channel ( last_out_ch, cur_out_ch, out_offset, output_count,
                            output_ch, act_min, act_max, bias_data, mult_ptr, shift_ptr, buffer,
                            out_ptr);
                
                // reset the buffer and flag
                buffer = (q31_t *)(ctx->buf);
                memset(buffer, 0, sizeof(q31_t) * output_count);
                mat_flag = 0;
            }

            if (double_flag && !(last_val == 0 && cur_val == 0)) {
                // use SIMD instructions to compute two val (last and cur val)
                // in depthwise, out_ch replace in_ch
                arm_nn_dw_convolve_s8_double_sparse(dw_conv_params,
                        stride_x_size, stride_y_size,
                        in_ptr,
                        input_x, input_y,
                        output_x, output_y,
                        input_ch,
                        last_out_ch, last_h,
                        last_w, last_val,
                        cur_out_ch, cur_h,
                        cur_w, cur_val,
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
            arm_nn_dw_convolve_s8_single_sparse(dw_conv_params,
                        stride_x_size, stride_y_size,
                        in_ptr,
                        input_x, input_y,
                        output_x, output_y,
                        input_ch,
                        cur_out_ch, cur_h,
                        cur_w, cur_val,
                        buffer);
        }
        
        double_flag = 0;

        // start to output finally  
        arm_nn_output_per_channel ( cur_out_ch, output_ch, out_offset, output_count,
                    output_ch, act_min, act_max, bias_data, mult_ptr, shift_ptr, buffer,
                    out_ptr);

        buffer = (q31_t *)(ctx->buf);
        memset(buffer, 0, sizeof(q31_t) * output_count);
    }

    return ARM_MATH_SUCCESS;
}
