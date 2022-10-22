#include "func.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

int32_t arm_fc_s8_sparse_get_buffer_size(const cmsis_nn_dims *output_dims)
{
    return 0;
}

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
                            const int32_t input_count,
                            const int32_t block)
{
    (void)bias_dims;
	int32_t *buffer = (int32_t*)ctx->buf;
    
    // conv information
    const int32_t batch = input_dims->n;
    const int32_t input_ch = filter_dims->n;
    const int32_t output_ch = output_dims->c;
    
    const int32_t in_offset = fc_params->input_offset;
    const int32_t out_offset = fc_params->output_offset;
    const int32_t act_max = fc_params->activation.max;
    const int32_t act_min = fc_params->activation.min;
    
    // used for SIMD acceleration in conv
    q31_t input_15x2, kernel_15x2;
    q31_t offset_q15x2 = __PKHBT(in_offset, in_offset, 16);
    const q7_t *filter_ptr, *in_ptr_0, *in_ptr_1, *end_ptr;
    q7_t *out_ptr;

    // used for decode and storage
    int32_t two_count = 0;
    q7_t last_val = 0, cur_val = 0;
    int32_t last_out_ch = 0, cur_out_ch = 0;
    int32_t last_in_ch = 0, cur_in_ch = 0;
    int32_t mat_flag = 0;
    printf("add cur_out_ch:%d %d\r\n",cur_out_ch, &cur_out_ch);
    // used for output and requant
    int32_t mult = quant_params->multiplier;
    int32_t shift = quant_params->shift;
    int32_t requant, bias = 0;
    int32_t block_cnt = 0;

    memset(buffer, 0, sizeof(int32_t) * batch);

    last_val = 0;
    last_out_ch = 0;
    last_in_ch = 0;

    two_count = 0;
    mat_flag = 0;
    block_cnt = 0;

    filter_ptr = filter_data;
    end_ptr = filter_ptr + input_count;

    while (filter_ptr < end_ptr) {
        // decode procedure
        cur_out_ch = last_out_ch;
        if (block_cnt == 0) {
            cur_in_ch = (*filter_ptr++) + last_in_ch + 128;
            cur_val = *filter_ptr++;

            if (cur_val == 0) {
                block_cnt = block - 1;
            }
        } else {
            cur_in_ch = last_in_ch + 1;
            cur_val = (*filter_ptr++);
        }

        if (++block_cnt >= block) {
            block_cnt = 0;
        }
        
        while (cur_in_ch >= input_ch) {
            ++cur_out_ch;
            cur_in_ch -= input_ch;
            mat_flag = 1;
        }
        
        if (mat_flag) {
            // change the output channel, last output channel conv is done
            if (two_count == 1 && last_val != 0) {
                // remain one val to conv (last_val)
                in_ptr_0 = input_data + last_in_ch;
                for (int32_t i_batch = 0; i_batch < batch; ++i_batch) {
                    buffer[i_batch] += ((*in_ptr_0 + in_offset) * last_val);
                    in_ptr_0 += input_ch;
                }
            }
            two_count = 0;

            // start to output
            // output step 1: output last out_channel (conv is done)
            bias_data?(bias = bias_data[last_out_ch]):(bias = 0);
            out_ptr = output_data + last_out_ch;

            for (int32_t i_batch = 0; i_batch < batch; ++i_batch) {
                requant = arm_nn_requantize(buffer[i_batch] + bias, mult, shift);
                requant += out_offset;
                requant = MAX(requant, act_min);
                requant = MIN(requant, act_max);
                *out_ptr = (q7_t)requant;
                out_ptr += output_ch;
            }

            // output step 2: output remaining empty channels (from last_out_ch to cur_out_ch)
            for (int32_t i_out_ch = last_out_ch + 1; i_out_ch < cur_out_ch; i_out_ch++) {
                bias_data?(bias=bias_data[i_out_ch]):(bias=0);
                out_ptr = output_data + i_out_ch;
                for (int32_t i_batch = 0; i_batch < batch; ++i_batch) {
                    requant = arm_nn_requantize(bias, mult, shift);
                    requant += out_offset;
                    requant = MAX(requant, act_min);
                    requant = MIN(requant, act_max);
                    *out_ptr = (q7_t)requant;
                    out_ptr += output_ch;
                }
            }

            // output step 3: reset the buffer and flag
            memset(buffer, 0, sizeof(int32_t) * batch);
            mat_flag = 0;
        }

        if (two_count && !(last_val == 0 && cur_val == 0)) {
            // use SIMD instructions to compute two val (last and cur val)
            kernel_15x2 = __PKHBT(last_val, cur_val, 16);
            in_ptr_0 = input_data + last_in_ch;
            in_ptr_1 = input_data + cur_in_ch;
            for (int32_t i_batch = 0; i_batch < batch; ++i_batch) {
                input_15x2 = __PKHBT(*in_ptr_0, *in_ptr_1, 16);
                input_15x2 = __QADD16(input_15x2, offset_q15x2);
                buffer[i_batch] = __SMLAD(kernel_15x2, input_15x2, buffer[i_batch]);
                in_ptr_0 += input_ch;
                in_ptr_1 += input_ch;
            }
        }

        two_count = 1 - two_count;                
        last_out_ch = cur_out_ch;
        last_in_ch = cur_in_ch;
        last_val = cur_val;
    }
    
    if (two_count == 1 && last_val != 0) {
        // remain one val to conv (last_val)
        in_ptr_0 = input_data + cur_in_ch;
        for (int i_batch = 0; i_batch < batch; ++i_batch) {
            buffer[i_batch] += ((*in_ptr_0 + in_offset) * cur_val);
            in_ptr_0 += input_ch;
        }
    }

    // start to output finally  
    // output step 1: output cur out_channel (conv is done)
    bias_data?(bias = bias_data[cur_out_ch]):(bias = 0);
    out_ptr = output_data + cur_out_ch;

    for (int i_batch = 0; i_batch < batch; ++i_batch) {
        requant = arm_nn_requantize(buffer[i_batch] + bias, mult, shift);
        requant += out_offset;
        requant = MAX(requant, act_min);
        requant = MIN(requant, act_max);
        *out_ptr = (q7_t)requant;
        out_ptr += output_ch;
    }

    // output step 2: output remaining empty channels (from cur_out_ch to output_ch)
    for (int32_t i_out_ch = cur_out_ch + 1; i_out_ch < output_ch; i_out_ch++) {
        bias_data?(bias=bias_data[i_out_ch]):(bias=0);
        out_ptr = output_data + i_out_ch;
        for (int i_batch = 0; i_batch < batch; ++i_batch) {
            requant = arm_nn_requantize(bias, mult, shift);
            requant += out_offset;
            requant = MAX(requant, act_min);
            requant = MIN(requant, act_max);
            *out_ptr = (q7_t)requant;
            out_ptr += output_ch;
        }
    }
    
    return ARM_MATH_SUCCESS;
}
