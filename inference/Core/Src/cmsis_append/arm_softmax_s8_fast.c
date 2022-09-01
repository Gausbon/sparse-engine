#include "arm_nnsupportfunctions.h"

#define ACCUM_BITS 12

__STATIC_FORCEINLINE int32_t get_exp_on_neg_mult_sat( int32_t *bit_map,
                     int32_t *exp_table,
                     int32_t diff,
                     int32_t mask,
                     int32_t mult)
{
    // bitmap: 32 * 8 = 256
    if (bit_map[(-diff) >> 5] & (1 << (-diff & 0x1f))) {
        return exp_table[-diff];
    } else {
        exp_table[-diff] = EXP_ON_NEG(MUL_SAT(diff * mask, mult));
        bit_map[(-diff) >> 5] |= (1 << (-diff & 0x1f));
        return exp_table[-diff];
    }
}


void arm_softmax_s8_fast(const cmsis_nn_context *ctx,
                    const int8_t *input,
                    const int32_t num_rows,
                    const int32_t row_size,
                    const int32_t mult,
                    const int32_t shift,
                    const int32_t diff_min,
                    int8_t *output)
{
    int32_t *buf = ctx->buf;
    int32_t *bit_map = buf;
    int32_t *exp_table = buf + 8;       // 256 / 32

    memset(bit_map, 0, 256);

    const int32_t mask = (1 << shift);

    int32_t col = 0;
    int32_t row_idx;

    for (row_idx = 0; row_idx < num_rows; ++row_idx)
    {
        // Find the maximum value in order to ensure numerical stability
        int8_t max = *input;

        for (col = 1; col < row_size; ++col)
        {
            max = MAX(max, input[col]);
        }

        int32_t diff = 0;
        int32_t sum = 0;

        for (col = 0; col < row_size; ++col)
        {
            diff = input[col] - max;
            if (diff >= diff_min)
            {
                sum += DIV_POW2(get_exp_on_neg_mult_sat(bit_map, exp_table, diff, mask, mult), ACCUM_BITS);
            }
        }

        const int32_t headroom = __CLZ(sum);
        const int32_t shifted_scale = ONE_OVER1((sum > 0 ? sum << headroom : 0) - (1 << 31));
        int32_t bits_over_unit;

        int8_t *output_s8 = (int8_t *)output + row_idx * row_size;

        bits_over_unit = ACCUM_BITS - headroom + 23;

        for (col = 0; col < row_size; ++col)
        {
            diff = input[col] - max;
            if (diff >= diff_min)
            {
                const int32_t res =
                    DIV_POW2(MUL_SAT(shifted_scale, get_exp_on_neg_mult_sat(bit_map, exp_table, diff, mask, mult))
                        , bits_over_unit) + NN_Q7_MIN;
                output_s8[col] = (int8_t)CLAMP(res, (int32_t)NN_Q7_MAX, (int32_t)NN_Q7_MIN);
            }
            else
            {
                output_s8[col] = NN_Q7_MIN;
            }
        }

        input += row_size;
    }
}
