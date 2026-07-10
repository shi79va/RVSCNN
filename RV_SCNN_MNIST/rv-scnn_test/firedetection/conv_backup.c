// //inside more than 4  , padding =1 , c_out more than 4  => correct
// //RELU is correct but increases the number of cycles
// ///The problem of Bias is solved!! ,,, Note::the value of bias should be negetive , because the hardware subtracts it internally!!

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../include/custom.h"

// -----------------------------------------------------------------------------
//
//----------Software reference for single conv + pool (like Base_conv_mp)-------
//
// -----------------------------------------------------------------------------
int *Base_conv_mp(int inside, int c_in, int c_out, int8_t *pInputs, int8_t *pFilters, int stride, int padding)
{
    int ksize = 3; // 3x3 kernel
    int conv_out = ((inside + 2 * padding - ksize) / stride) + 1;
    int oside = ((conv_out - 2) / 2) + 1; // first 2 is for pool_size   second 2 is for pool_stride
    // int oside = (inside - 2) / 2; // after 3x3 conv + 2x2 pool   ////original///
    printf("oside base = %d\n", oside);
    int *temp_buffer1 = (int *)calloc(oside * oside * c_out, sizeof(int));

    for (int x = 0; x < c_out; x++)
    {
        for (int i = 0; i < oside; i++)
        {
            for (int j = 0; j < oside; j++)
            {
                int a[4] = {0, 0, 0, 0};
                for (int m = 0; m < c_in; m++)
                {
                    for (int my = 0; my < 2; my++)
                    {
                        for (int mx = 0; mx < 2; mx++)
                        {
                            a[mx + my * 2] += (pFilters[x + m * 9 * c_out + 0 * c_out] * pInputs[m * inside * inside + 2 * j + 2 * i * inside + mx + my * inside]);
                            a[mx + my * 2] += (pFilters[x + m * 9 * c_out + 1 * c_out] * pInputs[m * inside * inside + 2 * j + 2 * i * inside + 1 + mx + my * inside]);
                            a[mx + my * 2] += (pFilters[x + m * 9 * c_out + 2 * c_out] * pInputs[m * inside * inside + 2 * j + 2 * i * inside + 2 + mx + my * inside]);

                            a[mx + my * 2] += (pFilters[x + m * 9 * c_out + 3 * c_out] * pInputs[m * inside * inside + 2 * j + 2 * i * inside + inside + mx + my * inside]);
                            a[mx + my * 2] += (pFilters[x + m * 9 * c_out + 4 * c_out] * pInputs[m * inside * inside + 2 * j + 2 * i * inside + inside + 1 + mx + my * inside]);
                            a[mx + my * 2] += (pFilters[x + m * 9 * c_out + 5 * c_out] * pInputs[m * inside * inside + 2 * j + 2 * i * inside + inside + 2 + mx + my * inside]);

                            a[mx + my * 2] += (pFilters[x + m * 9 * c_out + 6 * c_out] * pInputs[m * inside * inside + 2 * j + 2 * i * inside + 2 * inside + mx + my * inside]);
                            a[mx + my * 2] += (pFilters[x + m * 9 * c_out + 7 * c_out] * pInputs[m * inside * inside + 2 * j + 2 * i * inside + 2 * inside + 1 + mx + my * inside]);
                            a[mx + my * 2] += (pFilters[x + m * 9 * c_out + 8 * c_out] * pInputs[m * inside * inside + 2 * j + 2 * i * inside + 2 * inside + 2 + mx + my * inside]);
                        }
                    }
                }

                int b = 0;
                if (a[0] < a[1])
                    b = a[1];
                else
                    b = a[0];
                if (b < a[2])
                    b = a[2];
                if (b < a[3])
                    b = a[3];

                temp_buffer1[x * oside * oside + j + i * oside] = b;
            }
        }
    }

    return temp_buffer1;
}
int *Base_conv(int inside, int c_in, int c_out,
               int8_t *pInputs, int8_t *pFilters,
               int stride, int padding)
{
    int ksize = 3;
    int oside = ((inside + 2 * padding - ksize) / stride) + 1;

    int *poutputss = (int *)calloc(oside * oside * c_out, sizeof(int));

    for (int x = 0; x < c_out; x++)
    {
        for (int i = 0; i < oside; i++)
        {
            for (int j = 0; j < oside; j++)
            {

                int sum = 0;

                for (int m = 0; m < c_in; m++)
                {
                    for (int ky = 0; ky < 3; ky++)
                    {
                        for (int kx = 0; kx < 3; kx++)
                        {

                            // ----- FIXED PADDING HANDLING -----
                            int in_y = i * stride + ky - padding;
                            int in_x = j * stride + kx - padding;

                            int8_t inval = 0; // default padded value

                            if (in_y >= 0 && in_y < inside &&
                                in_x >= 0 && in_x < inside)
                            {
                                inval = pInputs[m * inside * inside +
                                                in_y * inside + in_x];
                            }

                            // original filter indexing preserved
                            int fidx = x + m * 9 * c_out + ky * 3 * c_out + kx * c_out;

                            sum += inval * pFilters[fidx];
                        }
                    }
                }

                poutputss[x * oside * oside + i * oside + j] = sum;
            }
        }
    }

    return poutputss;
}

// -----------------------------------------------------------------------------
// Requanize() function
// -----------------------------------------------------------------------------

static inline uint8_t requantize(int x, float requant_mult, int z_y, int qmin, int qmax)
{
    float yf = x * requant_mult;

    // round-to-nearest
    int y = (int)(yf + (yf >= 0 ? 0.5f : -0.5f));

    y += z_y;

    if (y < qmin)
        y = qmin;
    if (y > qmax)
        y = qmax;

    return (uint8_t)y;
}
// --------------------------------------------------------------------------------------
//
// ---------------------------Accelerator conv + RELU +  pool (RVSCNN)-------------------
//
// --------------------------------------------------------------------------------------

int8_t *RVSCNN_conv_mp(int inside, int c_in, int c_out, int8_t *pInputs, int8_t *pFilters, int stride, int padding, int *conv_bias, const float *requant_mult, int output_zero_point)
{
    int ksize = 3; // 3x3 kernel
    int padded_size = inside + 2 * padding;
    int conv_out = ((padded_size + 0 - ksize) / stride) + 1; // conv output size
    int pool_size = 2, pool_stride = 2;
    int oside = ((conv_out - pool_size) / pool_stride) + 1; // pooled output size

    // final accelerator output (after pooling)
    uint8_t *poutputs = (int8_t *)calloc((size_t)oside * oside * c_out, sizeof(int8_t));
    if (!poutputs)
    {
        fprintf(stderr, "OOM poutputs\n");
        exit(1);
    }

    // Optional writeback buffer (not required if accelerator returns via POOL_WB_INT)
    // keep for potential use / debugging
    int wb_elems = conv_out * conv_out;

    // Allocate padded input and copy original into center
    int8_t *padded_inputs = (int8_t *)calloc((size_t)c_in * padded_size * padded_size, sizeof(int8_t));
    if (!padded_inputs)
    {
        fprintf(stderr, "OOM padded_inputs\n");
        exit(1);
    }

    for (int m = 0; m < c_in; m++)
    {
        for (int y = 0; y < inside; y++)
        {
            for (int x = 0; x < inside; x++)
            {
                padded_inputs[m * padded_size * padded_size + (y + padding) * padded_size + (x + padding)] =
                    pInputs[m * inside * inside + y * inside + x];
            }
        }
    }

    // number of 4-channel groups (ceil)
    int chn_num = (c_out + 3) / 4;

    // number of tiles along each axis (we step tiles by pool_stride=2)
    int tile = oside; // equals 1 + (padded_size-4)/2 for our config

    printf("tile: %d\n", tile);
    int t1 = record();

    // Let accelerator know dimensions (use padded_size so internal memory/layouts align)
    L_SCNN(c_in, padded_size, conv_out * conv_out, 0);

    for (int g = 0; g < chn_num; g++)
    {
        // set mode for each of the 4 hardware channels
        L_MODE(0, 1, conv_bias[g * 4 + 0], 1);
        L_MODE(0, 2, conv_bias[g * 4 + 1], 1);
        L_MODE(0, 3, conv_bias[g * 4 + 2], 1);
        L_MODE(0, 4, conv_bias[g * 4 + 3], 1);
        // pack up to 4 output-channel filters into the contiguous buffer accelerator expects
        int packed_len = 9 * 4 * c_in;
        int8_t *packed_filt = (int8_t *)malloc((size_t)packed_len * sizeof(int8_t));
        if (!packed_filt)
        {
            fprintf(stderr, "OOM packed_filt\n");
            exit(1);
        }

        // Fill packed_filt. If the global out_ch >= c_out, fill zeros for that channel.
        for (int m_in = 0; m_in < c_in; m_in++)
        {
            for (int kidx = 0; kidx < 9; kidx++)
            {
                for (int ch4 = 0; ch4 < 4; ch4++)
                {
                    int out_ch = g * 4 + ch4; // global output channel index
                    int8_t val = 0;
                    if (out_ch < c_out)
                    {
                        val = pFilters[(m_in * 9 + kidx) * c_out + out_ch];
                    }
                    else
                    {
                        val = 0; // padding for missing channels
                    }
                    packed_filt[(m_in * 9 + kidx) * 4 + ch4] = val;
                }
            }
        }

        // For each tile (tile rows m, tile cols n), call accelerator on the 4x4 patch starting at (m*2, n*2)
        for (int m = 0; m < tile; m++)
        {
            for (int n = 0; n < tile; n++)
            {
                // input pointer: start of 4x4 patch at row = m*pool_stride, col = n*pool_stride
                // linear offset in padded_inputs (channel interleaving / accelerator expectations assumed same as before)
                int row_offset = m * pool_stride;
                int col_offset = n * pool_stride;
                int8_t *inptr = padded_inputs + (row_offset * padded_size + col_offset); // assumes inner-most dimension is width

                // Call accelerator for this 4-channel group and this tile's input patch
                SCNN4x4(packed_filt, inptr);
                ///-----RELU-----///
                for (int ch4 = 0; ch4 < c_out; ch4++)
                {
                    RELU(NULL, 16);
                }
                // After SCNN4x4, perform pooling via accelerator's POOL/POOL_WB_INT interface.
                // Request max-pool of 2x2
                POOL(2);

                // Retrieve pooled outputs for up to 4 channels in this group
                int pooled_val;
                for (int ch4 = 0; ch4 < 4; ch4++)
                {
                    int global_ch = g * 4 + ch4;
                    if (global_ch >= c_out)
                    {
                        // nothing to write for non-existent channel
                        // but we still need to call POOL_WB_INT only for valid indices the accelerator expects.
                        // accelerator's POOL_WB_INT takes index 1..4 — only call when channel exists.
                        break;
                    }
                    POOL_WB_INT(&pooled_val, ch4 + 1);
                    //-----REQUANTIZE()-----//
                    uint8_t qval = requantize(
                        pooled_val,
                        requant_mult[global_ch], // (s_x * s_w) / s_y
                        output_zero_point,
                        0, 255);
                    // write pooled value into final output at [global_ch, m, n]
                    poutputs[global_ch * oside * oside + m * oside + n] = qval;
                }
            }
        }

        free(packed_filt);
    }

    int t2 = record();
    printf("Accelerator cycle count: %d\n", t2 - t1);

    // cleanup
    free(padded_inputs);

    return poutputs;
}

//-----------------------------------------------------------------------------
//
// --------------------------------------FLATTEN-------------------------------
//
// -----------------------------------------------------------------------------
uint8_t *flatten_int8(
    const int8_t *input,
    int channels,
    int height,
    int width)
{
    int size = channels * height * width;
    uint8_t *output = malloc(size * sizeof(uint8_t));
    if (!output)
    {
        fprintf(stderr, "OOM in flatten_int8\n");
        exit(1);
    }

    memcpy(output, input, size * sizeof(uint8_t));
    return output;
}

//-----------------------------------------------------------------------------//
//
// -----------------------------FULLY CONNECTED--------------------------------//
//
// ----------------------------------------------------------------------------//

uint8_t *linear_layer_q(
    const uint8_t *input,
    int in_size,
    int out_size,
    int8_t *weights,
    int32_t *bias,
    float *requant_mult, // per-output-channel
    int output_zero_point)
{
    uint8_t *output = (uint8_t *)calloc(out_size, sizeof(uint8_t));
    if (!output)
    {
        fprintf(stderr, "OOM linear output\n");
        exit(1);
    }
    int input_zero_point = 0;
    for (int o = 0; o < out_size; o++)
    {
        int32_t sum = 0;

        for (int i = 0; i < in_size; i++)
        {
            int32_t x = (int32_t)input[i] - input_zero_point;
            int32_t w = (int32_t)weights[o * in_size + i];
            sum += x * w;
        }

        sum += bias[o]; // bias already in INT32 domain

        float yf = sum * requant_mult[o];

        // round-to-nearest
        int32_t y = (int32_t)(yf + (yf >= 0 ? 0.5f : -0.5f));

        y += output_zero_point;

        if (y < 0)
            y = 0;
        if (y > 255)
            y = 255;
        // if( 0<= y<=255) y=y;

        output[o] = (uint8_t)y;
    }

    return output;
}

//-----------------------------------------------------------------------------
// load input_matrix
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------//
//
// ---------------Main testbench (same structure as example)--------------------//
//
// -----------------------------------------------------------------------------//

int main()
{
    int inside = 224; // input size
    int c_in1 = 3;    // first conv input channels
    int c_out1 = 32;  // first conv output channels
    int stride1 = 1, padding1 = 0;

    //-----------------------------------------------------------------------------
    // load input_matrix
    // -----------------------------------------------------------------------------
    // Allocate input matrix
    int8_t *input_matrix = (int8_t *)malloc(inside * inside * c_in1 * sizeof(int8_t));
    float scale = 0.003921568859368563;
    int zero_point = -128;
    int ip[150528] = {}

  for (int i = 0; i < 150527; i++)
    {
        // Normalize to [0,1]
        // float x = ip[i] / 255.0f;

        // // Quantization
        // float qf = x / scale + zero_point;

        // int q;
        // if (qf >= 0.0f)
        //     q = (int)(qf + 0.5f);
        // else
        //     q = (int)(qf - 0.5f);

        // // Clamp to [0,255]
        // if (q < -128)
        //     q = -128;
        // if (-128 <= q <= 127)
        //     q = q;
        // if (q > 127)
        //     q = 127;

        // input_matrix[i] = (uint8_t)q;

        // printf("x=%d  x_norm=%.4f  q=%u\n",
        //        i, x, input_matrix[i]);
    }

    //-----------------------------------------------------------------------------//
    //
    // --------------------- FIRST CONVOLUTION LAYER (3→32)-------------------------//
    //
    // -----------------------------------------------------------------------------//

    int8_t *filter1 = (int8_t *)malloc(3 * 3 * c_in1 * c_out1 * sizeof(int8_t));
    int bias1[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int8_t K1[32][3][9] = {
        {{50, -68, 47, 13, -34, -54, 4, 53, 29},
         {-88, 62, 69, -78, 9, -27, -81, 51, -22},
         {-127, 59, -3, 28, -35, -44, 54, 85, -64}},
        {{-23, -91, 63, -107, -65, -30, -61, -31, 71},
         {-40, -24, -127, -5, -42, -16, -59, -39, 41},
         {-122, -124, 93, 75, 42, -71, 82, -26, -8}},
        {{89, -61, -77, 1, -97, 32, 85, 70, -106},
         {-107, 88, -33, 70, -74, -17, 81, -48, -3},
         {-92, 37, 59, 5, -98, 76, 60, -127, -91}},
        {{-118, -93, -37, 31, -82, -17, -49, 37, 97},
         {36, 82, 60, 53, -26, -57, 32, -56, -114},
         {-14, -15, -50, 95, 59, -10, 11, -127, -101}},
        {{87, -43, -6, 48, -61, 4, 32, -34, -45},
         {26, -21, -85, -95, -2, -102, -60, -113, 0},
         {78, 41, -127, -53, -126, -18, -103, 43, -48}},
        {{13, 21, 7, 81, -58, 74, 95, 57, 23},
         {-99, -11, -66, -107, 66, 45, -95, 8, -9},
         {-110, -94, -87, 41, 46, -24, -95, 72, 64}},
        {{32, -14, -25, -15, 80, 43, -126, -34, -38},
         {-33, 80, 57, -36, 108, 6, 13, 99, 41},
         {14, 85, -127, 41, -16, -63, -23, -96, 41}},
        {{-110, 10, -125, -100, -9, 27, -60, 78, -48},
         {-40, 10, -104, -41, -9, 90, 112, 127, 78},
         {47, 102, -77, -41, 65, 80, 9, -29, 36}},
        {{64, -78, 89, 18, -120, -44, 23, 62, 83},
         {-91, 93, -15, -27, 36, -106, -110, -21, 36},
         {-69, 62, -39, 78, -37, 85, -110, 61, 29}},

        {{-58, 73, -92, 6, 60, -112, -5, -66, -9},
         {92, -46, -63, -101, 66, 87, -8, 83, 21},
         {79, -127, -25, 59, 96, -24, 65, 33, -124}},

        {{-108, 80, 1, 28, -55, -17, 16, -46, -63},
         {-89, 40, -76, -21, 20, 36, 34, 32, -63},
         {39, 56, 42, -81, -127, 55, -101, 43, -92}},

        {{-127, -118, 4, 54, -29, -37, 45, 7, -9},
         {-24, -52, -85, -7, -33, -50, 25, 7, -9},
         {-34, 112, 14, 127, -18, -106, 25, 15, -116}},

        {{33, -47, -15, 42, -120, 38, 2, -96, 12},
         {4, 69, -45, 105, 5, 67, 72, -96, 12},
         {-127, -61, 30, -49, -85, 37, 43, 60, -90}},

        {{-117, -61, -29, -127, -34, 96, -55, -72, -55},
         {-93, 81, 60, -49, -85, 37, -84, 30, -57},
         {-3, 12, 55, -5, 30, 37, 63, 16, 60}},

        {{-61, 44, -13, -53, 86, 26, -14, -98, 80},
         {40, 63, -89, -5, 4, -69, 63, 49, -48},
         {-93, 114, 70, -5, 28, 17, 63, -59, -83}},

        {{-115, 53, -5, -28, -118, 40, 28, 73, 27},
         {53, 46, -60, 34, -37, 80, -106, -77, -35},
         {40, 32, 18, -49, 1, 13, -106, 81, 60}},

        {{4, -45, 72, 82, -76, 46, 70, -58, 33},
         {-19, -21, 52, -120, -30, 41, -42, 83, -127},
         {111, -15, -73, 8, 84, -61, -42, -24, 35}},

        {{-53, -23, 50, -7, -45, 90, -3, -127, -87},
         {23, -66, -52, -7, -90, 30, -45, -24, -27},
         {-36, 25, -83, -40, -98, 37, -45, -28, -35}},

        {{36, 35, -103, 102, 101, 51, 76, 96, -80},
         {93, -127, -76, -34, -98, 47, -105, -28, 60},
         {-69, -71, -49, 8, 55, 13, 81, 16, 60}},

        {{5, -32, 54, 54, -55, 25, 88, 38, 106},
         {-55, -85, -57, -40, -98, 37, -104, 16, -34},
         {75, -64, -83, -40, -18, 37, 81, 16, 60}},

        {{46, 28, -20, 76, -98, -127, -118, 40, 66},
         {90, 71, -53, 34, 21, 13, 96, 49, -57},
         {18, -6, -49, -70, -24, 102, -104, -85, 17}},

        {{22, 23, 49, -17, 82, 102, 22, -69, 66},
         {23, -35, 106, 121, 69, -127, -104, -85, -101},
         {75, 80, -4, 58, 69, -127, 71, 53, -23}},

        {{-7, 14, -5, 3, -47, 54, -94, 27, 40},
         {14, -21, -127, 24, 77, 14, 71, 53, -23},
         {-59, -105, 122, 24, -1, 77, 71, -57, -54}},

        {{21, -107, 73, -91, 81, 85, 85, -42, 13},
         {-107, -18, 106, -127, 81, 0, 21, -57, -54},
         {-24, -86, -56, -127, 0, 1, 21, -57, -54}},

        {{-79, 34, 11, 75, -98, -25, -45, 81, 24},
         {34, -58, -21, -70, -98, -49, -49, 81, 24},
         {-127, 52, -110, 58, -102, 31, -73, -49, 66}},

        {{60, -35, 48, 74, -127, -59, -69, -73, -73},
         {-35, 58, 6, 58, -102, 98, 18, -49, 18},
         {4, 52, 76, 99, -119, 96, -54, 89, 0}},

        {{-94, -70, -44, 83, 88, 60, 98, -54, 0},
         {-70, -97, 88, 83, -119, -74, 0, 89, -127},
         {111, -15, -127, 99, -50, -127, -54, 80, 50}},

        {{-15, -55, 29, -34, 78, -37, -103, -57, 50},
         {-55, -7, 98, 46, 77, 96, -54, -57, -83},
         {-36, 25, -91, 10, 25, 3, -57, 80, 17}},

        {{69, 65, -25, 72, -67, -18, 37, 3, -83},
         {65, -5, -63, -69, 4, -20, -68, 3, 17},
         {-7, -23, -54, 20, 54, -37, -37, -57, -127}},

        {{61, 19, -99, 71, -118, 63, -127, -68, 17},
         {19, 52, -63, -127, 4, -20, -68, -57, 17},
         {75, 80, -23, 20, 54, -37, -37, -57, -127}}};
    int idx = 0;
    // for (int in = 0; in < c_in1; in++) {
    for (int i = 0; i < 9; i++)
    {
        for (int out = 0; out < 32; out++)
        {
            filter1[idx++] = K1[out][0][i];
        }
    }
    //}
    float requant_mult1[32] = {
        0.0015924613474258612, 0.0017561850964863653, 0.0017547084945501825, 0.001672219630259483, 0.001578975925482409, 0.001511123620779429, 0.0014749320952245468, 0.0015879653425605932, 0.0016673684320219974, 0.0017073300197836005, 0.001680455449720699, 0.0016987829091966598, 0.0015681903188547917, 0.0018924609536272062, 0.0018624736451505492, 0.0016049005477235547, 0.0015837256645911883, 0.0017124410017867518, 0.0017223577241894847, 0.0016529664210503847, 0.0016166969430814538, 0.0015820354281012045, 0.0015201731873921783, 0.0016236568318985284, 0.0015707236858896307, 0.0016075839429078268, 0.001533341268687751, 0.0015808530057852235, 0.0016150775724224236, 0.0017763298324175553, 0.0012955549483238868, 0.0015139978351768049};
    int output_zero_point = 0;
    // First layer convolution
    int8_t *out1 = RVSCNN_conv_mp(inside, c_in1, c_out1, input_matrix, filter1, stride1, padding1, bias1, requant_mult1, output_zero_point);

    // Compute first layer output size after pooling
    int conv_out1 = ((inside + 2 * padding1 - 3) / stride1) + 1;
    int oside1 = ((conv_out1 - 2) / 2) + 1;

    printf("L1 output side = %d (expected 14)\n", oside1);

    printf("First Conv Output (3→32):\n");
    for (int c = 0; c < c_out1; c++)
    {
        printf("Channel %d:\n", c);
        for (int i = 0; i < oside1; i++)
        {
            for (int j = 0; j < oside1; j++)
            {
                printf("%4d ", out1[c * oside1 * oside1 + i * oside1 + j]);
            }
            printf("\n");
        }
    }

    //-----------------------------------------------------------------------------//
    //
    // --------------------- SECOND CONVOLUTION LAYER (32→64)-----------------------//
    //
    // ----------------------------------------------------------------------------//
    int c_in2 = c_out1;
    int c_out2 = 64;
    int stride2 = 1, padding2 = 0;

    int8_t *filter2 = (int8_t *)malloc(3 * 3 * c_in2 * c_out2 * sizeof(int8_t));
    int bias2[64] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    int8_t K2[64][32][9] = {
        {{31, -28, -1, 44, 27, 16, -33, -12, -38},
         {-72, -88, -31, 92, -31, -21, -50, -53, -26},
         {-81, 24, -41, 54, 25, 82, -75, 59, 91},
         {-24, 35, 84, -50, 26, -60, 17, 32, -11},
         {55, 89, 90, 75, 72, 36, 92, 37, 92},
         {69, 89, 78, 72, -53, 11, -42, -70, 66},
         {18, -87, -36, 29, -82, -42, -79, -46, -101},
         {-96, -45, -17, -104, -88, 38, 18, -76, -13},
         {-86, -18, -28, -3, 57, -25, 34, 87, -13},
  .....

int idx1 = 0;
    for (int in = 0; in < 64; in++)
    {
        for (int i = 0; i < 9; i++)
        {
            for (int out = 0; out < 128; out++)
            {
                filter3[idx1++] = K3[out][in][i];
            }
        }
    }
    float requant_mult3[128] = {
        0.0011562642375260673, 0.0011639085208257662, 0.001256795825330879, 0.001059487145486816, 0.0011389425433974843, 0.0011150597526179435, 0.0012698603896248808, 0.0015625858270713966, 0.0011648715049580012, 0.0011006054426331044, 0.001195705496442068, 0.00127401999344607, 0.0010608942821800412, 0.0012387570933538336, 0.0010828691776975934, 0.0010402879903504234, 0.0012266881639465506, 0.0011564873368067028, 0.0011354098681623838, 0.0012303348184473024, 0.0011953487251430768, 0.0011135035136542438, 0.0014902984206450237, 0.0014625026543281605, 0.0012462636978846267, 0.0012668097179559681, 0.001249605413093505, 0.001234286412239928, 0.0012963045170061396, 0.0010941774212504204, 0.0011835876335280656, 0.0012360091444764904, 0.0014711062560095912, 0.001291501105096544, 0.0012429230909259009, 0.0011351848933814908, 0.0013003611388131874, 0.001152303266232154, 0.0013087730132182904, 0.0012015550964953243, 0.0013272881222607508, 0.0011195016192278924, 0.0012844534013787662, 0.001103560890038941, 0.0013853876249888672, 0.0012916687918695703, 0.0010914972461323837, 0.0011862101796381887, 0.0013899293193625243, 0.0012140807102153184, 0.0014232936164440479, 0.0013035288587481728, 0.0011166776273401071, 0.0012221583193245207, 0.0011142317192542395, 0.0010658405731092569, 0.0011791871984238058, 0.0013137413839005379, 0.0010880510996591653, 0.0010643965231609625, 0.001064110168371641, 0.0013583497312760736, 0.0012040451640872557, 0.0012978930657242761};

    // Second layer convolution
    int8_t *out3 = RVSCNN_conv_mp(oside2, c_in3, c_out3, out2, filter3, stride3, padding3, bias3, requant_mult3, output_zero_point);

    // Compute second layer output size
    int conv_out3 = ((oside2 + 2 * padding3 - 3) / stride3) + 1;
    int oside3 = ((conv_out3 - 2) / 2) + 1;
    printf("L2 output side = %d (expected 7)\n", oside3);

    printf("Second Conv Output (64→128):\n");
    for (int c = 0; c < c_out3; c++)
    {
        printf("Channel %d:\n", c);
        for (int i = 0; i < oside3; i++)
        {
            for (int j = 0; j < oside3; j++)
            {
                printf("%4d   ", out2[c * oside3 * oside3 + i * oside3 + j]);
            }
            printf("\n");
        }
    }

        uint8_t *flat = flatten_int8(out3, c_out3, oside3, oside3);

        int flatten_size = c_out3 * oside3 * oside3;

        // printf(" Flatten size (size=%d):\n", flatten_size1);
        // printf("Flattened Output (size = %d):\n", c_out3 * oside3 * oside3);
        // for (int i = 0; i < c_out3 * oside3 * oside3; i++)
        // {
        //     printf("%4d, ", flat[i]);
        // }
    //     printf("\n");

    //     //-----------------------------------------------------------------------------//
    //     //
    //     // --------------------------- FULLY CONNECTED --------------------------------//
    //     //                             LAYER 5 86528->128
    //     // -------------------------------software-------------------------------------//

        int output_zero_pointfc1 = 70;
        int fc1_in = flatten_size;
        int fc1_out = 128;
        int32_t *fc1_bias = (int32_t *)malloc(fc1_out * sizeof(int32_t));
        int8_t *fc1_output = (int8_t *)malloc(64 * sizeof(int8_t));

        for (int i = 0; i < 128; i++)
            fc1_bias[i] = 0;

            int8_t FC1[86528][128] = {
}





  


float fc_requant_mult = 1.2760520752480434e-05f;

int input_zero_point  = -128;
int output_zero_point = -128;

for (int n = 0; n < 128; n++)
{
    int32_t sum = 0;

    for (int i = 0; i < 86528; i++)
    {
        sum += (int32_t)FC1[i][n] *
               ((int32_t)flat[i] - input_zero_point);
    }

    // bias (if available)
    sum += FC1_bias[n];

    // Requantize
    float yf = sum * fc_requant_mult;
    int32_t y = (int32_t)(yf + (yf >= 0 ? 0.5f : -0.5f));

    // Add output zero point
    y += output_zero_point;

    // ReLU
    if (y < output_zero_point)
        y = output_zero_point;

    // Clamp to int8
    if (y < -128)
        y = -128;

    if (y > 127)
        y = 127;

    fc1_output[n] = (int8_t)y;
}
    //     //-----------------------------------------------------------------------------//
    //     //
    //     // --------------------------- FULLY CONNECTED --------------------------------//
    //     //                             LAYER 4 784->64
    //     // ------------------------------accelerator-------------------------------------//
    // #define K 4
    // #define N 4

    //     int8_t x[K] = {1, 2, 3, 4};

    //     int8_t W[K][N] = {
    //         {1, 2, 3, 4},
    //         {5, 6, 7, 8},
    //         {9, 10, 11, 12},
    //         {1, 1, 1, 1}};
    //     for (int n = 0; n < N; n++)
    //     {
    //         int sum = 0;
    //         for (int k = 0; k < K; k++)
    //             sum += x[k] * W[k][n];
    //         printf("SW[%d] = %d\n", n, sum);
    //     }

    //     int8_t A_fc[4][4];
    //     for (int r = 0; r < 4; r++)
    //         for (int k = 0; k < 4; k++)
    //             A_fc[r][k] = x[k];

    //     int8_t B4x4[16];
    //     for (int k = 0; k < 4; k++)
    //         for (int n = 0; n < 4; n++)
    //             B4x4[k * 4 + n] = W[k][n];

    //     int fc_out[4] = {0};

    //     L_SCNN(4, 4, 4, 0);

    //     /* HARD RESET — DO THIS ONCE */
    //     L_MODE(0, 1, 0, 1);

    //     /* EXACTLY ONE MAC */
    //     SCNN4x4(&A_fc[0][0], B4x4);

    //     /* WRITE BACK ONCE */
    //     SCNN_WB_INT(fc_out);

    //     printf("%d %d %d %d\n",
    //            fc_out[0], fc_out[1], fc_out[2], fc_out[3]);
    //     printf("%d %d %d %d\n",
    //            fc_out[0], fc_out[4], fc_out[8], fc_out[12]);

    //     printf("FC output:\n");
    //     for (int n = 0; n < 4; n++)
    //         printf("HW[%d] = %d\n", n, fc_out[n]);

    //     ////////////////////////////////////

    //     //-----------------------------------------------------------------------------//
    //     //
    //     // --------------------------- FULLY CONNECTED --------------------------------//
    //     //                             LAYER 5 128->128
    //     // ----------------------------------------------------------------------------//

        int fc2_in = fc1_out; // 128
        int fc2_out = 128;
        int32_t *fc2_bias = (int32_t *)malloc(fc2_out * sizeof(int32_t));

        // Initialize weights/bias
        for (int i = 0; i < fc2_out; i++)
            fc2_bias[i] = 0;
        // int* fc2_output = linear_layer(fc1_output, fc2_in, fc2_out, fc2_weights, fc2_bias);
        int8_t *fc2_output = (int8_t *)malloc(10 * sizeof(int8_t));
        const float fc2_requant_mult = 0.001505410750309097f;
        int output_zero_pointfc2 = -128;
        int8_t FC2[64][10] = {
            {-91, 87, 12, -128, 57, -128, -19, 14, 48, -1},
            {-36, 34, -41, 13, 3, -24, 59, -128, -3, 8},
            {-81, 8, -40, 57, 43, -36, -119, -66, 50, 36},
            {2, -32, -26, -8, 6, 19, -35, -21, 6, 6},
            {-7, 32, 23, 23, -18, -28, -13, -39, 12, 29},
            {-3, -35, -53, -34, 16, -47, 58, 60, 17, 47},
            {36, -19, 37, -19, -114, -8, -37, -3, -17, 34},
            {19, -69, 44, -84, 21, 29, -7, -77, 6, 41},
            {23, -87, -38, 13, -51, 38, -10, 26, 23, -9},
            {46, 63, -128, -27, 13, 0, -54, 3, 17, 15},
            {-128, 52, -7, 35, 42, 47, -100, 44, -23, 37},
            {34, -72, 66, -93, 39, -2, -92, 76, -31, 38},
            {-2, -127, 35, -49, -44, 11, 36, -9, 24, 39},
            {10, 43, -9, 42, -69, 36, -67, 54, -75, -23},
            {-95, -118, -39, -19, 12, 13, -103, 58, -7, 40},
            {6, -27, 27, -61, 24, -29, -42, -23, 16, 32},
            {20, -16, 20, -29, -16, -25, -10, 24, 0, -32},
            {-77, 4, 9, 0, -4, 21, -4, 0, -11, 12},
            {-5, -93, 38, 60, 5, 16, 51, 31, 22, 7},
            {35, -55, 27, 22, -128, 31, 14, -45, -2, -61},
            {-53, -5, -66, 20, 56, 21, -76, 68, -22, -16},
            {-45, -23, -83, -109, -3, 40, 56, 2, 31, -2},
            {-49, 18, 32, -14, -28, -51, 54, 12, 23, -128},
            {12, -18, 36, -30, -20, -75, -66, 62, 46, -104},
            {36, -72, -79, 30, -1, -45, 37, -82, -50, 23},
            {31, -15, -11, 62, -83, 56, -33, -55, -43, 4},
            {8, 27, 23, 7, 61, 26, 2, 53, -127, 22},
            {-38, 87, -83, -100, -10, -12, 8, 31, -44, 67},
            {6, -21, -3, 6, 27, -29, -31, 30, -2, 27},
            {-16, 51, 47, -24, -13, 1, 29, -87, 30, -41},
            {-109, -34, 11, 25, 14, 42, 25, 20, -19, -45},
            {-25, -12, -8, -29, 17, -19, 25, -23, -4, 4},
            {61, 23, 15, -64, -19, -84, 12, 65, 22, 19},
            {55, 2, -81, 10, 25, -98, -30, -40, 22, 40},
            {-36, -67, 21, 62, 19, -60, -128, -9, 10, 34},
            {17, 29, -7, -58, -39, 18, -26, -79, -1, 53},
            {0, -26, -10, -31, -18, 7, -27, -6, 5, 30},
            {-2, -86, 35, -4, -4, 20, 33, -93, 30, 20},
            {43, 17, -43, -100, 67, 24, 27, 1, -16, -32},
            {46, -81, -6, -24, 24, 46, 49, -59, -12, -13},
            {-11, 10, 23, 34, 26, 17, -35, 19, 9, -29},
            {-32, 33, 27, -57, -42, -26, -95, 50, 9, 5},
            {12, -21, 25, 36, -29, 10, -11, 21, 14, -23},
            {-73, -11, 58, -15, 19, 36, -4, -114, 26, -67},
            {3, -48, -15, -40, -8, -4, 10, -16, -1, -10},
            {-97, -103, -106, 18, 29, 58, -3, -92, 24, 18},
            {20, -20, 68, -38, 16, -107, -75, 20, 27, -35},
            {41, -59, -38, 57, -27, -11, -19, -14, 21, 23},
            {33, -42, -52, 20, -54, -19, -6, -101, 28, -24},
            {37, 56, -28, -95, 6, 12, 31, 6, 29, -69},
            {5, -5, 34, 26, 1, -125, 35, 30, -56, -56},
            {30, -8, 2, -13, 25, -34, 16, -17, -13, 19},
            {18, -27, 4, 3, -22, -31, 10, -1, 7, -4},
            {35, 80, 51, 4, -34, -97, -3, 61, -56, -103},
            {-81, 38, 33, 66, -45, -10, -99, 12, 7, 19},
            {50, 23, -67, 65, -117, 7, -34, 29, -17, 8},
            {-31, 31, -32, -15, 7, 24, 12, 8, -11, 21},
            {-18, -15, 4, 12, 24, -33, -21, 37, 6, 23},
            {-50, 70, 14, 4, -70, 72, 24, 13, -56, -23},
            {-28, -25, 16, 20, 16, -6, 25, 0, 21, -28},
            {22, 19, -28, 1, 3, -6, 5, -24, 23, -27},
            {-75, 49, 31, 26, -19, -65, -79, 73, -48, -47},
            {-53, 44, -32, 47, -74, 20, 21, -60, 14, 2},
            {42, -49, 21, 9, 16, 9, 13, -111, 8, -43},
        };

        for (int n = 0; n < 10; n++)
        {
            int32_t sum = 0;
            for (int i = 0; i < 64; i++)
            {
                sum += (int32_t)FC2[i][n] * ((int32_t)fc1_output[i] - 70);
            }
            // Requantize
            float yf = sum * fc_requant_mult2[n];
            int32_t y = (int32_t)(yf + (yf >= 0 ? 0.5f : -0.5f));

            // add output zero point
            y += output_zero_pointfc2;
            // clamp to uint8
            if (y < 0)
                y = 0;
            if (y > 255)
                y = 255;

            fc2_output[n] = (uint8_t)y;
        }

        printf("Final FC Output (size=%d):\n", fc2_out);
        for (int i = 0; i < fc2_out; i++)
        {
            printf("%4d ", fc2_output[i]);
        }
        printf("\n");

    int *base_conv = Base_conv(inside, c_in1, c_out1, input_matrix, filter1, stride1, padding1);
    // int *base_conv1 = Base_conv(oside1, c_in2, c_out2, out1_i8, filter2, stride2, padding2);

    printf("Convolution Output Matrices (c_out1=%d, oside1=%d):\n", c_out1, conv_out1);

    for (int oc = 0; oc < 1; oc++)
    {
        printf("Channel %d:\n", oc);
        for (int i = 0; i < conv_out1; i++)
        {
            for (int j = 0; j < conv_out1; j++)
            {
                int idx = oc * (conv_out1 * conv_out1) + i * conv_out1 + j;
                printf("%4d ", (int)base_conv[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // free memory
    free(input_matrix);
    free(filter1);
    free(filter2);
    free(filter3);
    free(out1);
    free(out2);
    free(out3);

    // free(fc2_output);
    return 0;
}
