#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void load_fc_weights_from_file(int8_t fc2_weight[10][64], const char* filename) {
    FILE *f = fopen(filename, "r");
    if (f == NULL) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        exit(1);
    }
    char line[100];
    while (fgets(line, sizeof(line), f)) {
        int i, j, val;
        if (sscanf(line, "fc2.weight[%d][%d] = %d", &i, &j, &val) == 3) {
            if (i >= 0 && i < 10 && j >= 0 && j < 64) {
                fc2_weight[i][j] = (int8_t)val;
            }
        }
    }
    fclose(f);
}

// Example usage with printing:
int main() {
    int8_t fc2_weight[10][64];
    load_fc_weights_from_file(fc2_weight, "weights_int8(bias_int8)_matrix_conv1.txt");

    // Print in C array initialization format
    printf("int8_t fc2_weight[10][64] = {\n");
    for (int i = 0; i < 10; i++) {
        printf("  {");
        for (int j = 0; j < 64; j++) {
            printf("%d", fc2_weight[i][j]);
            if (j < 63) printf(", ");
            if ((j + 1) % 16 == 0) printf("\n   ");  // Line break for readability every 16 values
        }
        printf("}");
        if (i < 9) printf(",");
        printf("\n");
    }
    printf("};\n");

    // Note: No biases were found in the file, so if needed, initialize them separately (e.g., to 0).
    return 0;
}