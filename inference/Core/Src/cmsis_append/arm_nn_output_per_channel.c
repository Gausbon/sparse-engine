#include "arm_nn_tables.h"
#include "arm_nnsupportfunctions.h"

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
                                q7_t *output_data
) {
    int32_t requant;
    const q31_t *input_ptr = &input_data[0];
    
    // output step 1: output last out_channel (conv is done)
    q7_t *output_ptr = &output_data[start_channel];
    
    int32_t mult = mult_data[start_channel];
    int32_t shift = shift_data[start_channel];
    
    if (bias_data) {
        int32_t bias = bias_data[start_channel];
        for (int32_t i_buf = 0; i_buf < output_count; i_buf++) {
            requant = arm_nn_requantize((*input_ptr) + bias, mult, shift);
            requant += out_offset;
            requant = MAX(requant, act_min);
            requant = MIN(requant, act_max);
            *output_ptr = (q7_t)requant;
            output_ptr += output_ch;
            input_ptr++;
        }
    } else {
        for (int32_t i_buf = 0; i_buf < output_count; i_buf++) {
            requant = arm_nn_requantize((*input_ptr), mult, shift);
            requant += out_offset;
            requant = MAX(requant, act_min);
            requant = MIN(requant, act_max);
            *output_ptr = (q7_t)requant;
            output_ptr += output_ch;
            input_ptr++;
        }
    }
    

    // output step 2: output remaining empty channels (from start_channel to end_channel)
    if (bias_data) {
        for (int32_t i_out_ch = start_channel + 1; i_out_ch < end_channel; i_out_ch++) {
            int32_t bias = bias_data[i_out_ch];
            mult = mult_data[i_out_ch];
            shift = shift_data[i_out_ch];
            output_ptr = &output_data[i_out_ch];
            
            requant = arm_nn_requantize(bias, mult, shift);
            requant += out_offset;
            requant = MAX(requant, act_min);
            requant = MIN(requant, act_max);
            
            for (int32_t i_buf = 0; i_buf < output_count; i_buf++) {
                *output_ptr = (q7_t)requant;
                output_ptr += output_ch;
            }
        }
    } else {
        for (int32_t i_out_ch = start_channel + 1; i_out_ch < end_channel; i_out_ch++) {
            mult = mult_data[i_out_ch];
            shift = shift_data[i_out_ch];
            output_ptr = &output_data[i_out_ch];

            requant = MAX(out_offset, act_min);
            requant = MIN(requant, act_max);
            
            for (int32_t i_buf = 0; i_buf < output_count; i_buf++) {
                *output_ptr = (q7_t)requant;
                output_ptr += output_ch;
            }
        }
    }
}

void arm_nn_output_per_channel_CHW (   const int32_t start_channel,
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
                                q7_t *output_data
) {
    int32_t requant;
    const q31_t *input_ptr = &input_data[0];
    
    // output step 1: output last out_channel (conv is done)
    q7_t *output_ptr = output_data + start_channel * output_count;
    
    int32_t mult = mult_data[start_channel];
    int32_t shift = shift_data[start_channel];
    
    if (bias_data) {
        int32_t bias = bias_data[start_channel];
        for (int32_t i_buf = 0; i_buf < output_count; i_buf++) {
            requant = arm_nn_requantize((*input_ptr++) + bias, mult, shift);
            requant += out_offset;
            requant = MAX(requant, act_min);
            requant = MIN(requant, act_max);
            *output_ptr++ = (q7_t)requant;
        }
    } else {
        for (int32_t i_buf = 0; i_buf < output_count; i_buf++) {
            requant = arm_nn_requantize((*input_ptr++), mult, shift);
            requant += out_offset;
            requant = MAX(requant, act_min);
            requant = MIN(requant, act_max);
            *output_ptr++ = (q7_t)requant;
        }
    }
    

    // output step 2: output remaining empty channels (from start_channel to end_channel)
    if (bias_data) {
        for (int32_t i_out_ch = start_channel + 1; i_out_ch < end_channel; i_out_ch++) {
            int32_t bias = bias_data[i_out_ch];
            mult = mult_data[i_out_ch];
            shift = shift_data[i_out_ch];
            
            requant = arm_nn_requantize(bias, mult, shift);
            requant += out_offset;
            requant = MAX(requant, act_min);
            requant = MIN(requant, act_max);
            
            for (int32_t i_buf = 0; i_buf < output_count; i_buf++) {
                *output_ptr++ = (q7_t)requant;
            }
        }
    } else {
        for (int32_t i_out_ch = start_channel + 1; i_out_ch < end_channel; i_out_ch++) {
            mult = mult_data[i_out_ch];
            shift = shift_data[i_out_ch];

            requant = MAX(out_offset, act_min);
            requant = MIN(requant, act_max);
            
            for (int32_t i_buf = 0; i_buf < output_count; i_buf++) {
                *output_ptr++ = (q7_t)requant;
            }
        }
    }
}
