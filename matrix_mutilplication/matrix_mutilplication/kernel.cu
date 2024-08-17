
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEBUG 1

#define M 512
#define L 512
#define N 512


/*********************************** utils functions ***********************************/
// 比较结果
bool compare_result(int* a, int* b) {
	for (int i = 0; i < M * N; ++i) {
		if (a[i] != b[i]) {
			printf("Error: %d %d %d\n", i, a[i], b[i]);
			return false;
		}
	}
	return true;
}


/*********************************** c model ***********************************/
// 朴素矩阵乘法
void matrix_mul_c(int* a, int* b, int* c) {

    for(int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < L; ++k) {
                sum += a[i * L + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

// 循环交换，a 的 一个元素与 b 的一行元素相成
void matrix_mul_c_v2(int* a, int* b, int* c) {
    // set c to 0
    for (int i = 0; i < M * N; ++i) {
        c[i] = 0;
    }

    for (int i = 0; i < M; ++i) {
        // 交换循环
        for (int k = 0; k < L; ++k) {
            int tmp = a[i * L + k]; // 取 a 的一个元素
            for (int j = 0; j < N; ++j) { // 与 b 的一行元素相乘
                c[i * N + j] += tmp * b[k * N + j];
            }
        }
    }
}

// 转置矩阵乘法，a 的一行元素与 b 的一行元素相乘
void matrix_mul_c_v3(int* a, int* b, int* c) {

    int* b_T = (int*)malloc(N * L * sizeof(int));

#if DEBUG == 1
    clock_t start, end;
    start = clock();
#endif

    // 转置 b
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < L; ++j) {
            b_T[i * L + j] = b[j * N + i];
        }
    }

#if DEBUG == 1
	end = clock();
    printf(" v3 transpose time: %fs\n", (double)(end - start) / CLOCKS_PER_SEC);
    start = clock();
#endif

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int tmp = 0;
            for (int k = 0; k < L; ++k) { // a 的一行元素与 b 的一行元素相乘
                tmp += a[i * N + k] * b_T[j * N + k];
            }
            c[i * N + j] = tmp;
        }
    }

#if DEBUG == 1  
    end = clock();
    printf(" v3 matrix multi time: %fs\n", (double)(end - start) / CLOCKS_PER_SEC);
#endif

    free(b_T);
}

/*********************************** cuda model ***********************************/





/*********************************** main ***********************************/
int main()
{
    clock_t start, end;

    int* a = (int*)malloc(M * L * sizeof(int));
    int* b = (int*)malloc(L * N * sizeof(int));
    int* c = (int*)malloc(M * N * sizeof(int));
    
    // init data
    start = clock();
    for (int i = 0; i < M * L; ++i) {
		a[i] = (int)(rand() & 0xFF);
        b[i] = (int)(rand() & 0xFF);
        c[i] = 0;
	}    
    end = clock();
    printf("init time: %fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    // matrix multiplication on cpu
    start = clock();
    matrix_mul_c(a, b, c);
	end = clock();
    printf("cpu time: %fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    // matrix multiplication on cpu v2
    int* c_v2 = (int*)malloc(M * N * sizeof(int));
    start = clock();
    matrix_mul_c_v2(a, b, c_v2);
	end = clock();
    printf("cpu v2 time: %fs\n", (double)(end - start) / CLOCKS_PER_SEC);
    compare_result(c, c_v2);

    // matrix multiplication on cpu v3
    int* c_v3 = (int*)malloc(M * N * sizeof(int));
    start = clock();
    matrix_mul_c_v3(a, b, c_v3);
    end = clock();
    printf("cpu v3 time: %fs\n", (double)(end - start) / CLOCKS_PER_SEC);
    compare_result(c, c_v3);




    // free
    free(a);
    free(b);
    free(c);
    free(c_v2);
    free(c_v3);

    return 0;
}

