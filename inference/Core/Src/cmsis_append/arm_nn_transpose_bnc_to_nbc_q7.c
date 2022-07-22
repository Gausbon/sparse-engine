#include "arm_nnsupportfunctions.h"

arm_cmsis_nn_status arm_nn_transpose_bnc_to_nbc_q7(const int32_t dim_b, 
                                            const int32_t dim_n, 
                                            const int32_t dim_c, 
                                            const q7_t *input_section, 
                                            q7_t *output_section)
{
    const q7_t *input_start = input_section;
    for (int32_t i_n = 0; i_n < dim_n; ++i_n) {
        const q7_t *input_ptr = input_start;
        q7_t *output_ptr = output_section;
        for (int32_t i_b = 0; i_b < dim_b; ++i_b) {
            memcpy(output_ptr, input_ptr, sizeof(q7_t) * dim_c);
            output_ptr += dim_c;
            input_ptr += (dim_c * dim_n);
        }
        input_start += dim_c;
    }
    
    return ARM_CMSIS_NN_SUCCESS;
}


