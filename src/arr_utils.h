//
// Created by peter on 29.10.18.
//

#ifndef PROJECT_SRC_ARR_UTILS_H
#define PROJECT_SRC_ARR_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void print_array(double* arr, int size)
{
    for (int i = 0; i < size; ++i) {
        printf("%f, ",arr[i]);
    }
    printf("\n");
}

void print_array_int(int* arr, int size)
{
    for (int i = 0; i < size; ++i) {
        printf("%d, ",arr[i]);
    }
    printf("\n");
}

void free_mat(double** mat, int W)
{
    for (int i = 0; i < W; ++i) {
        free(mat[i]);
    }
    free(mat);
}

void sign_matrix(double **mat, int W, int H)
{
    for (int i = 0; i < W; ++i) {
        for (int j = 0; j < H; ++j)
        {
            if(mat[i][j] > 0)
            {
                mat[i][j] = 1.0;
            }
            if(mat[i][j] < 0)
            {
                mat[i][j] = -1.0;
            }
            if(mat[i][j] == 0)
            {
                mat[i][j] = 0.0;
            }
        }
    }
}


double sigm(double x)
{
    return 1.0 / (1.0 + exp(x));
}

double pow2(int d)
{
    return (double)(1<<d);
}

static int double_cmp_up(const void * a, const void * b)
{
    if (*(double*)a > *(double*)b) return -1;
    else if (*(double*)a < *(double*)b) return 1;
    else return 0;
}

static int double_cmp_down(const void * a, const void * b)
{
    if (*(double*)a > *(double*)b) return 1;
    else if (*(double*)a < *(double*)b) return -1;
    else return 0;
}



struct __val_idx_struct
{
    double value;
    int index;
};

int __val_idx_cmp_up(const void *a, const void *b)
{
    struct __val_idx_struct *a1 = (struct __val_idx_struct *)a;
    struct __val_idx_struct *a2 = (struct __val_idx_struct*)b;
    if((*a1).value>(*a2).value)return -1;
    else if((*a1).value < (*a2).value)return 1;
    else return 0;
}

int __val_idx_cmp_down(const void *a, const void *b)
{
    struct __val_idx_struct *a1 = (struct __val_idx_struct *)a;
    struct __val_idx_struct *a2 = (struct __val_idx_struct*)b;
    if((*a1).value>(*a2).value)return 1;
    else if((*a1).value < (*a2).value)return -1;
    else return 0;
}

int* argsort(double* arr, int size)
{
    struct __val_idx_struct* objects = (struct __val_idx_struct*)malloc(size * sizeof(struct __val_idx_struct));
    for(int i=0;i<size;i++)
    {
        objects[i].value=arr[i];
        objects[i].index=i;
    }

    //sort objects array according to value maybe using qsort
    qsort(objects, size, sizeof(objects[0]), __val_idx_cmp_up);

    int* sort_idxs = (int*)malloc(size* sizeof(int));
    for (int i = 0; i < size; ++i) {
        sort_idxs[i] = objects[i].index;
    }

    free(objects);
    return sort_idxs;
}


void sort_arr(double* arr, int size)
{
    printf("SORT:\n");
    print_array(arr, size);
    qsort(arr, size, sizeof(double), double_cmp_up);
}


double* get_by_idx(double* arr, int* idxs, int size)
{
    double* res = (double*)malloc(size * sizeof(double));
    for (int i = 0; i < size; ++i) {
        res[i] = arr[idxs[i]];
    }
    return res;
}

void swap_arr_int(int* y, int i, int j)
{
    int buf = y[i];
    y[i] = y[j];
    y[j] = buf;
}

void swap_arr_double(double* y, int i, int j)
{
    double buf = y[i];
    y[i] = y[j];
    y[j] = buf;
}


double* copy_arr(double* arr, int size)
{
    double* res = (double*)malloc(size * sizeof(double));
    for (int i = 0; i < size; ++i) {
        res[i] = arr[i];
    }
    return res;
}


#endif //PROJECT_SRC_ARR_UTILS_H
