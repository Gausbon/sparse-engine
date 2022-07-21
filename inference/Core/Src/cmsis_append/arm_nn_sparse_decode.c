#include "arm_nn_tables.h"
#include "arm_nnsupportfunctions.h"

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
                                    q7_t *cur_val)
{
    const q7_t *filter_ptr = *filter_data;
    
    *cur_out_ch = last_out_ch;
    *cur_h = last_h;
    *cur_w = last_w;
    *cur_in_ch = *(filter_ptr++) + last_in_ch + 128;
    *cur_val = *(filter_ptr++);
    while (*cur_in_ch >= input_ch) {
        *cur_w += (*cur_in_ch / input_ch);
        *cur_in_ch = *cur_in_ch % input_ch;
        while (*cur_w >= kernel_x) {
            *cur_h += (*cur_w / kernel_x);
            *cur_w = *cur_w % kernel_x;
            while (*cur_h >= kernel_y) {
                *cur_out_ch += (*cur_h / kernel_y);
                *cur_h = *cur_h % kernel_y;
                *mat_flag = 1;
            }
        }
    }
    *counter -= 2;
}

void arm_nn_sparse_decode_2d(    const int32_t last_in_ch,
                                    const int32_t last_out_ch,
                                    const int32_t input_ch,
                                    const q7_t **filter_data,
                                    int32_t *cur_in_ch,
                                    int32_t *cur_out_ch,
                                    int32_t *mat_flag,
                                    int32_t *counter,
                                    q7_t *cur_val) 
{
    const q7_t *filter_ptr = *filter_data;
    
    *cur_out_ch = last_out_ch;
    *cur_in_ch = *(filter_ptr++) + last_in_ch + 128;
    *cur_val = *(filter_ptr++);
    while (*cur_in_ch >= input_ch) {
        *cur_out_ch += (*cur_in_ch / input_ch);
        *cur_in_ch = *cur_in_ch % input_ch;
        *mat_flag = 1;
    }
    *counter -= 2;
}
