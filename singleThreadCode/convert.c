#include "hpl-ai.h"
void convert_double_to_float(double *src, int ldsrc, float *dst,
                             int lddst, int m, int n) {
    int i, j;
    for (j = 0; j < n; ++j) {
        for (i = 0; i < m; ++i){
            *dst = (float)(*src);
            dst++;
            src++;
        }
        dst += lddst - m;
        src += ldsrc - m;
    }
    return;
}
void convert_float_to_double(float *src, int ldsrc, double *dst,
                             int lddst, int m, int n) {
    int i, j;
    for (j = 0; j < n; ++j) {
        for (i = 0; i < m; ++i){
            *dst = (double)(*src);
            dst++;
            src++;
        }
        dst += lddst - m;
        src += ldsrc - m;
    }
    return;
}
/*

#define S(i, j) *HPLAI_INDEX2D(src, (i), (j), ldsrc)
#define D(i, j) *HPLAI_INDEX2D(dst, (i), (j), lddst)

void convert_double_to_float(double *src, int ldsrc, float *dst,
                             int lddst, int m, int n) {
    int i, j;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            D(i, j) = (float)S(i, j);
        }
    }
    return;
}

void convert_float_to_double(float *src, int ldsrc, double *dst,
                             int lddst, int m, int n) {
    int i, j;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            D(i, j) = (double)S(i, j);
        }
    }
    return;
}
*/