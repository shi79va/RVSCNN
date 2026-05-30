#include <stdio.h>
#include <stdint.h>
#include "../include/custom.h"
#include <stdio.h>
#include <stdint.h>

int RELU(int8_t *addr, int len); // prototype from above

int main() {
    int8_t data[8] = { -3, 1, -2, 7, 0, -1, 5, -8 };
    int rc = RELU(data, 8);
    printf("RELU returned: %d\n", rc);
    for (int i = 0; i < 8; ++i) printf("%d ", data[i]);
    printf("\n");
    return 0;
}

