#include "arm_nnfunctions.h"
#include "stdio.h"

int result_check(const cmsis_nn_context *ctx,
                          const q7_t *section,
                          const int32_t class) {
    printf("Inference result:\r\n");
    for(int i = 0; i < 10; i++){
        printf("%d ",section[i]);
    }
    printf("\r\n");
    int32_t *buffer = (int32_t*) ctx->buf;
    for(int i = 0; i < 10; i++){
        buffer[i] = i;
    }

    for (int i = 1; i < 10; i++) {
			int j = i;
			while (j) {
				if (section[i] > section[j-1]) {
					buffer[i]--;
					buffer[j - 1]++;
				}
				j--;
			}
		}

    printf("Index sort:\r\n");
    for(int i = 0; i < 10; i++){
        printf("%d ",buffer[i]);
    }
    printf("\r\nclass:%d\r\n",class);

    if (buffer[class] < 1) {
      printf("Top 1 check: True\r\n");
    } else {
      printf("Top 1 check: False\r\n");
    }

    if (buffer[class] < 5) {
      printf("Top 5 check: True\r\n");
    } else {
      printf("Top 5 check: False\r\n");
    }

    return 0;
}
