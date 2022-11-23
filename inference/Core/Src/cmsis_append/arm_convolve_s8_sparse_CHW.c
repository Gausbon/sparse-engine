#include "func.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "stdio.h"
void arm_nn_convolve_s8_double_sparse_1x1_CHW ( const cmsis_nn_conv_params *conv_params,
                                    const q7_t *input_data,
                                    const int32_t input_x,
                                    const int32_t input_y,
                                    const int32_t output_x,
                                    const int32_t output_y,
                                    const int32_t input_ch,
                                    const int32_t in_ch_0,
                                    const int32_t in_ch_1,
                                    const int32_t h,
                                    const int32_t w,
                                    const int32_t ker_val_0,
                                    const int32_t ker_val_1,
                                    const int32_t input_size,
                                    q31_t *buffer)
{
    const int32_t pad_x = conv_params->padding.w;
    const int32_t pad_y = conv_params->padding.h;
    const int32_t in_offset = conv_params->input_offset;
    const int32_t dis = (in_ch_1 - in_ch_0) * input_x * input_y;

    int32_t k_y, k_x;
    q7_t input_0, input_1;
    const q7_t *input_start;
    const q7_t *input_ptr_0, *input_ptr_1;
    q31_t kernel_15x2 = __PKHBT(ker_val_0, ker_val_1, 16);
    q31_t offset_15x2 = __PKHBT(in_offset, in_offset, 16);

    input_start = input_data + in_ch_0 * input_size;

    if ((k_y = h - pad_y) > 0) {
        input_start += (k_y * input_x);
    }
    if ((k_x = w - pad_x) > 0) {
        input_start += k_x;
    }

    for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++) {
        k_x = w - pad_x;
        input_ptr_0 = input_start;
        input_ptr_1 = input_start + dis;
        // not both padding
        if ((k_y >= 0 && k_y < input_y)) {
            for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++) {
                if (k_y >= 0 && k_y < input_y && k_x >= 0 && k_x < input_x) {
                    input_0 = *(input_ptr_0++);
                    input_1 = *(input_ptr_1++);
                } else {
                    input_0 = -in_offset;
                    input_1 = -in_offset;
                }
                
                if (k_y == 0 && in_ch_0 == 4 && in_ch_1 == 5) {
                    // printf("input: %d %d, ker: %d, %d\r\n",input_0, input_1, ker_val_0, ker_val_1);
                }
                int32_t input_15x2 = __PKHBT(input_0, input_1, 16);
                input_15x2 = __QADD16(input_15x2, offset_15x2);
                (*buffer) = __SMLAD(kernel_15x2, input_15x2, (*buffer));
                
                ++buffer;
                ++k_x;
            }

            if (k_y >= 0 && k_y < input_y) {
                input_start += input_x;
            }
        } else {
            buffer += input_x;
        }
        ++k_y;
    }
}

void arm_nn_convolve_s8_double_sparse_1x1_CHW_debug ( const cmsis_nn_conv_params *conv_params,
                                    const q7_t *input_data,
                                    const int32_t input_x,
                                    const int32_t input_y,
                                    const int32_t output_x,
                                    const int32_t output_y,
                                    const int32_t input_ch,
                                    const int32_t in_ch_0,
                                    const int32_t in_ch_1,
                                    const int32_t h,
                                    const int32_t w,
                                    const int32_t ker_val_0,
                                    const int32_t ker_val_1,
                                    const int32_t input_size,
                                    q31_t *buffer)
{
    const int32_t pad_x = conv_params->padding.w;
    const int32_t pad_y = conv_params->padding.h;
    const int32_t in_offset = conv_params->input_offset;
    const int32_t dis = (in_ch_1 - in_ch_0) * input_x * input_y;

    int32_t k_y, k_x;
    q7_t input_0, input_1;
    const q7_t *input_start;
    const q7_t *input_ptr_0, *input_ptr_1;
    q31_t kernel_15x2 = __PKHBT(ker_val_0, ker_val_1, 16);
    q31_t offset_15x2 = __PKHBT(in_offset, in_offset, 16);

    input_start = input_data + in_ch_0 * input_size;

    if ((k_y = h - pad_y) > 0) {
        input_start += (k_y * input_x);
    }
    if ((k_x = w - pad_x) > 0) {
        input_start += k_x;
    }

    for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++) {
        k_x = w - pad_x;
        input_ptr_0 = input_start;
        input_ptr_1 = input_start + dis;
        // not both padding
        if ((k_y >= 0 && k_y < input_y)) {
            for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++) {
                if (k_y >= 0 && k_y < input_y && k_x >= 0 && k_x < input_x) {
                    input_0 = *(input_ptr_0++);
                    input_1 = *(input_ptr_1++);
                } else {
                    input_0 = -in_offset;
                    input_1 = -in_offset;
                }
                
                if (k_y == 0 && in_ch_0 == 4 && in_ch_1 == 5) {
                    // printf("input: %d %d, ker: %d, %d\r\n",input_0, input_1, ker_val_0, ker_val_1);
                }
                int32_t input_15x2 = __PKHBT(input_0, input_1, 16);
                input_15x2 = __QADD16(input_15x2, offset_15x2);
                (*buffer) = __SMLAD(kernel_15x2, input_15x2, (*buffer));
                
                ++buffer;
                ++k_x;
            }

            if (k_y >= 0 && k_y < input_y) {
                input_start += input_x;
            }
        } else {
            buffer += input_x;
        }
        ++k_y;
    }
}

arm_status arm_convolve_s8_sparse_1x1_CHW (const cmsis_nn_context *ctx,
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

    if (conv_params->stride.w != 1 || conv_params->stride.h != 1 || conv_params->dilation.w != 1
        || conv_params->dilation.h != 1) {
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

    const int32_t out_offset = conv_params->output_offset;
    const int32_t act_min = conv_params->activation.min;
    const int32_t act_max = conv_params->activation.max;

    q31_t *buffer = (q31_t *)(ctx->buf);
    q31_t *dec_buf = buffer + (arm_convolve_s8_sparse_get_buffer_size(output_dims) >> 2);
    q31_t *dec_ptr = dec_buf;
    q31_t *dec_end_ptr = dec_buf;
    q31_t *dec_buf_end = buffer + ((ctx->size) >> 2);

    q7_t val_1 = 0, val_2 = 0;
    int32_t pos_1[4] = {0};     // in_ch, w, h, out_ch
    int32_t pos_2[4] = {0};

    q31_t *mult_ptr = quant_params->multiplier;
    q31_t *shift_ptr = quant_params->shift;

    int32_t input_size = input_x * input_y;
    int32_t output_size = output_x * output_y;

    int32_t block_cnt = 0;
    int32_t res = 0;

    for (int32_t i_batch = 0; i_batch < batch; ++i_batch) {
        memset(buffer, 0, sizeof(q31_t) * output_size);
        
        const q7_t *filter_ptr = filter_data;
        const q7_t *end_ptr = filter_ptr + input_count;
        const q7_t *in_ptr = &input_data[i_batch * input_size * input_ch];
        q7_t *out_ptr = &output_data[i_batch * output_size * output_ch];

        block_cnt = 0;
        
        while (1) {
            // decode procedure
            if (dec_ptr < dec_end_ptr) {
                if (block_cnt == 0) {
                    memcpy(pos_1, dec_ptr, 16);
                    dec_ptr += 4;
                    val_1 = (q7_t) (*dec_ptr++);
                    if (pos_2[3] != pos_1[3]) {
                        arm_nn_output_per_channel_CHW (pos_2[3], pos_1[3], out_offset, output_size,
                            output_ch, act_min, act_max, bias_data, mult_ptr, shift_ptr, buffer,
                            out_ptr);
                        buffer = (q31_t *)(ctx->buf);
                        memset(buffer, 0, sizeof(q31_t) * output_size);
                    }
                    memcpy(pos_2, pos_1, 16);
                    ++pos_2[0];
                    val_2 = (q7_t) (*dec_ptr++);
                } else {
                    pos_1[0] += 2;
                    val_1 = (q7_t) (*dec_ptr++);

                    pos_2[0] += 2;
                    val_2 = (q7_t) (*dec_ptr++);
                }
                block_cnt = (block_cnt + 2) % block;
            } else {
                dec_ptr = dec_buf;
                //printf("start\r\n");
                dec_end_ptr = arm_nn_decode_4d (dec_buf, dec_buf_end,
                        &filter_ptr, end_ptr,
                        pos_2, &res,
                        input_ch, kernel_x,
                        kernel_y, block);
                if (dec_end_ptr == dec_buf) {
                    break;
                }
                continue;
            }
            //printf("%d %d %d %d %d\r\n",last_in_ch, last_w, last_h, last_out_ch, val_1);
            //printf("%d %d %d %d %d\r\n",cur_in_ch, cur_w, cur_h, cur_out_ch, val_2);

            arm_nn_convolve_s8_double_sparse_1x1_CHW(conv_params,
                        in_ptr,
                        input_x, input_y,
                        output_x, output_y,
                        input_ch,
                        pos_1[0], pos_2[0],
                        pos_1[2], pos_1[1], 
                        val_1, val_2,
                        input_size,
                        buffer);
        }

        // start to output finally  
        arm_nn_output_per_channel_CHW ( pos_2[3], output_ch, out_offset, output_size,
                    output_ch, act_min, act_max, bias_data, mult_ptr, shift_ptr, buffer,
                    out_ptr);

        buffer = (q31_t *)(ctx->buf);
        memset(buffer, 0, sizeof(q31_t) * output_size);
    }

    return ARM_MATH_SUCCESS;
}

arm_status arm_convolve_s8_sparse_1x1_CHW_debug (const cmsis_nn_context *ctx,
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

    if (conv_params->stride.w != 1 || conv_params->stride.h != 1 || conv_params->dilation.w != 1
        || conv_params->dilation.h != 1) {
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

    const int32_t out_offset = conv_params->output_offset;
    const int32_t act_min = conv_params->activation.min;
    const int32_t act_max = conv_params->activation.max;

    q31_t *buffer = (q31_t *)(ctx->buf);
    q31_t *dec_buf = buffer + (arm_convolve_s8_sparse_get_buffer_size(output_dims) >> 2);
    q31_t *dec_ptr = dec_buf;
    q31_t *dec_end_ptr = dec_buf;
    q31_t *dec_buf_end = buffer + ((ctx->size) >> 2);

    q7_t val_1 = 0, val_2 = 0;
    int32_t pos_1[4] = {0};     // in_ch, w, h, out_ch
    int32_t pos_2[4] = {0};

    q31_t *mult_ptr = quant_params->multiplier;
    q31_t *shift_ptr = quant_params->shift;

    int32_t input_size = input_x * input_y;
    int32_t output_size = output_x * output_y;

    int32_t block_cnt = 0;
    int32_t res = 0;

    for (int32_t i_batch = 0; i_batch < batch; ++i_batch) {
        memset(buffer, 0, sizeof(q31_t) * output_size);
        
        const q7_t *filter_ptr = filter_data;
        const q7_t *end_ptr = filter_ptr + input_count;
        const q7_t *in_ptr = &input_data[i_batch * input_size * input_ch];
        q7_t *out_ptr = &output_data[i_batch * output_size * output_ch];

        block_cnt = 0;
        
        while (1) {
            // decode procedure
            if (dec_ptr < dec_end_ptr) {
                if (block_cnt == 0) {
                    memcpy(pos_1, dec_ptr, 16);
                    dec_ptr += 4;
                    val_1 = (q7_t) (*dec_ptr++);
                    if (pos_2[3] != pos_1[3]) {
                        printf("output:%d %d\r\n",pos_2[3], pos_1[3]);
                        arm_nn_output_per_channel_CHW (pos_2[3], pos_1[3], out_offset, output_size,
                            output_ch, act_min, act_max, bias_data, mult_ptr, shift_ptr, buffer,
                            out_ptr);
                        
                        buffer = (q31_t *)(ctx->buf);
                        memset(buffer, 0, sizeof(q31_t) * output_size);
                    }
                    memcpy(pos_2, pos_1, 16);
                    ++pos_2[0];
                    val_2 = (q7_t) (*dec_ptr++);
                } else {
                    pos_1[0] += 2;
                    val_1 = (q7_t) (*dec_ptr++);

                    pos_2[0] += 2;
                    val_2 = (q7_t) (*dec_ptr++);
                }
                block_cnt = (block_cnt + 2) % block;
            } else {
                dec_ptr = dec_buf;
                //printf("start\r\n");
                dec_end_ptr = arm_nn_decode_4d (dec_buf, dec_buf_end,
                        &filter_ptr, end_ptr,
                        pos_2, &res,
                        input_ch, kernel_x,
                        kernel_y, block);
                if (dec_end_ptr == dec_buf) {
                    break;
                }
                continue;
            }

            if (pos_1[1] != pos_2[1] || pos_1[2] != pos_2[2] || pos_1[3] != pos_2[3]) {
                printf("warning\r\n");
            }
            if (pos_1[3] == 10) {
                printf("%d %d %d %d %d\r\n",pos_1[0], pos_1[1], pos_1[2], pos_1[3], val_1);
                printf("%d %d %d %d %d\r\n",pos_2[0], pos_2[1], pos_2[2], pos_2[3], val_2);
            }

            arm_nn_convolve_s8_double_sparse_1x1_CHW_debug (conv_params,
                        in_ptr,
                        input_x, input_y,
                        output_x, output_y,
                        input_ch,
                        pos_1[0], pos_2[0],
                        pos_1[2], pos_1[1], 
                        val_1, val_2,
                        input_size,
                        buffer);
        }

        // start to output finally
        printf("output:%d %d\r\n",pos_2[3], output_ch);
        arm_nn_output_per_channel_CHW ( pos_2[3], output_ch, out_offset, output_size,
                    output_ch, act_min, act_max, bias_data, mult_ptr, shift_ptr, buffer,
                    out_ptr);

        buffer = (q31_t *)(ctx->buf);
        memset(buffer, 0, sizeof(q31_t) * output_size);
    }

    return ARM_MATH_SUCCESS;
}