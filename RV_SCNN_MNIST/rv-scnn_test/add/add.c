#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
//#include "../include/custom.h"   // (optional, for consistency with your format)

// -----------------------------------------------------------------------------
// Simple function: add two integers (software reference)
// -----------------------------------------------------------------------------
int Base_add(int a, int b) {
    int result = a + b;
    return result;
}

// -----------------------------------------------------------------------------
// Accelerator-based addition (example style matching RVSCNN format)
// -----------------------------------------------------------------------------
// int RVSCNN_add(int a, int b) {
//     int result = 0;
    
//     // Record start time
//     int t1 = record();

//     // --- Example: if we had a custom instruction for addition ---
//     // For demonstration, just use normal addition here
//     result = a + b;

//     // Record end time
//     int t2 = record();

//     printf("Accelerator cycle count: %d\n", t2 - t1);
//     return result;
// }

// -----------------------------------------------------------------------------
// Error check function
// -----------------------------------------------------------------------------
// int error_check(int base_result, int accel_result) {
//     return abs(base_result - accel_result);
// }

// -----------------------------------------------------------------------------
// Main testbench (same structure as previous RVSCNN test codes)
// -----------------------------------------------------------------------------
int main() {
    int a = 7;
    int b = 5;

    // Software reference
    int base_result = Base_add(a, b);

    // Accelerator result
    //int accel_result = RVSCNN_add(a, b);

    // Error check
    //int error = error_check(base_result, accel_result);

    // printf("Inputs: a=%d, b=%d\n", a, b);
    // printf("Base result: %d\n", base_result);
    // printf("Accelerator result: %d\n", accel_result);
    // printf("Error: %d\n", error);

    // //

    return 0;
}
