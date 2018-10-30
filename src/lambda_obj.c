#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "arr_utils.h"







double calculate_dcg(double *y, int size)
{
    //printf("dcg\n");
    //printf("dcg_y_size: %d\n", size);
    //printf("dcg_y_arr: ");
    //print_array(y, size);

    double dcg = 0.0;
    for(int i =0; i< size; ++i)
    {
        //printf("dcg i: %d\n", i);
        dcg += (pow2((int)y[i]) - 1)/log(2.0 + i);
    }
    return dcg;
}


//double calculate_ideal_dcg(double *y, int size)
//{
//    double idcg = 0.0;
//    double* y_sorted = (double*)malloc(size*sizeof(double));
//    for (int i = 0; i < size; ++i) {
//        y_sorted[i] = y[i];
//    }
//
//    qsort(y_sorted, size, sizeof(double), (__compar_fn_t) double_cmp);
//
//    for(int i =0; i< size; ++i)
//    {
//        idcg += (pow2((int)y_sorted[i]) - 1)/log(2.0 + i);
//    }
//
//
//    free(y_sorted);
//    return idcg;
//}




//double calculate_ndcg(double *y, double* f, int size)
//{
//    double _dcg = calculate_dcg(y, size);
//    double _idcg = calculate_ideal_dcg(y, size);
//    //printf("ndcg_ideal_dcg\n");
//
//    return _dcg/_idcg;
//}

double calculate_ndcg(double* y, double* f, int size)
{

    int* f_sorted_idxs = argsort(f, size);
    int* y_sorted_idxs = argsort(y, size);

    double* y_by_f = get_by_idx(y, f_sorted_idxs, size);
    double* sorted_y = get_by_idx(y, y_sorted_idxs, size);

    double dcg = calculate_dcg(y_by_f, size);
    double ideal_dcg = calculate_dcg(sorted_y, size);


    free(f_sorted_idxs);
    free(y_sorted_idxs);
    free(y_by_f);
    free(sorted_y);

    if(dcg == 0)
    {
        return 0;
    }

    return dcg/ideal_dcg;
}

double delta_ndcg(double* _y, double* _f,  int i, int j, int size)
{
    double* y_copy = copy_arr(_y, size);
    double* f_copy = copy_arr(_f, size);

    double ndcg_1 = calculate_ndcg(y_copy, f_copy, size);

    double buf = y_copy[i];
    y_copy[i] = y_copy[j];
    y_copy[j] = buf;

    double  ndcg_2 = calculate_ndcg(y_copy, f_copy, size);

    free(y_copy);
    free(f_copy);

    return fabs(ndcg_1 - ndcg_2);
}



double** pairwise_diff(double* x, int size)
{
    //printf("call_pairwise_diff\n");
    double** res = (double**)malloc(size*sizeof(double*));
    for (int i = 0; i < size; ++i) {
        res[i] = (double*)malloc(size*sizeof(double));
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            res[i][j] = x[i] - x[j];
        }
    }
    return res;
}





extern void LambdaRankObjective(double* Y,
                                    double* F,
                                    double sigma,
                                    int size,
                                    int* group, int group_size,
                                    double* grad, double* hess)
{

    //printf("SIZE: %d\n", size);

//    double* grad = (double*)malloc(size*sizeof(double));
//    double* hess = (double*)malloc(size*sizeof(double));

    //printf("C_HELLO\n");
//    print_array(Y, size);
//    print_array(F, size);
//    print_array_int(group, group_size);


    int group_start = 0;
    int group_len =  0;

    int count_grad_zero = 0;
    int count_grad_nonzero = 0;

    for (int group_i = 0; group_i < group_size; group_i++)
    {
        //printf("gruop_i: %d \n", group_i);

        group_start += group_len;
        group_len = group[group_i];

        double* group_Y = Y + group_start;
        double* group_F = F + group_start;

        double** pairwise_s = pairwise_diff(group_Y, group_len);
        sign_matrix(pairwise_s, group_len, group_len);

        double** pairwise_f = pairwise_diff(group_F, group_len);

        for (int i = 0; i < group_len; ++i)
        {
            //printf("i: %d\n", i);
            double cur_grad = 0;
            double cur_hess = 0;

            for (int j = 0; j < group_len; ++j) {
                //printf("i, j: %d, %d \n", i, j);
                double _delta_ndcg = delta_ndcg(group_Y, group_F,  i, j, group_len);
                //printf("i, j: %d, %d delta_ndcg \n", i, j);
                double pij = sigm( sigma * fabs(pairwise_f[i][j]) );
                //printf("i, j: %d, %d pij \n", i, j);
                cur_grad += pairwise_s[i][j] *(-sigma * _delta_ndcg * pij);
                //printf("i, j: %d, %d cur_grad \n", i, j);

                //if(pairwise_s[i][j] > 0) {
                cur_hess += fabs(pairwise_s[i][j]) * sigma * sigma * _delta_ndcg * pij * (1.0 - pij);
                    //printf("i, j: %d, %d cur_hess \n", i, j);
                //}
            }

            if(cur_grad == 0)
                count_grad_zero++;
            else
                count_grad_nonzero++;

            grad[group_start + i] = cur_grad;
            //printf("i: %d\n set_grad", i);
            hess[group_start + i] = cur_hess;
            //printf("i: %d\n set_hess", i);
        }

        free_mat(pairwise_s, group_len);
        //printf("gruop_i: %d free_mat_s\n", group_i);
        free_mat(pairwise_f, group_len);
        //printf("gruop_i: %d free_mat_f\n", group_i);
    }

//    printf("NONZERO_GRAD: %d\n", count_grad_nonzero);
//    printf("ZERO_GRAD: %d\n", count_grad_zero);
//    printf("GRAD: \n");
//    for (int i = 0; i < size; ++i) {
//        printf("grad %d: %f \n", i, grad[i]);
//    }
//
//    printf("HESS: \n");
//    for (int i = 0; i < size; ++i) {
//        printf("hess %d: %f \n", i, hess[i]);
//    }


//    double** res = (double**)malloc(2 * sizeof(double*));
//    res[0] = grad;
//    res[1] = hess;
//
//    printf("RETURN\n");
//    Y[0] = 9998.0;
//    F[0] = 5672.0;
//
//    return res;
}


int count_lines(char* filename)
{
    FILE* fp =  fopen(filename,"r");
    int lines = 0;
    int ch=0;
    while(!feof(fp))
    {
        ch = fgetc(fp);
        if(ch == '\n')
        {
            lines++;
        }
    }

    fclose(fp);
    return lines;
}

void read_file(double* arr, char* filename)
{
    FILE* fp =  fopen(filename,"r");
    int idx = 0;
    double val =0;


    while(fscanf(fp, "%lf", &val) > 0)
    {
        arr[idx] = val;
        idx++;
    }

    fclose(fp);

}

void read_file_int(int* arr, char* filename)
{
    FILE* fp =  fopen(filename,"r");
    int idx = 0;
    int val =0;


    while(fscanf(fp, "%d", &val) > 0)
    {
        arr[idx] = val;
        idx++;
    }

    fclose(fp);

}

//void test_lambda()
//{
//    int Y_size = count_lines("Y_test.txt");
//    int F_size = count_lines("F_test.txt");
//    int group_size = count_lines("group_test.txt");
//
//    double* Y = (double*)malloc(Y_size*sizeof(double));
//    double* F = (double*)malloc(F_size*sizeof(double));
//    int* group = (int*)malloc(group_size*sizeof(int));
//
//    read_file(Y, "Y_test.txt");
//    read_file(F, "F_test.txt");
//    read_file_int(group, "group_test.txt");
//
//    double* grad = (double*)malloc(Y_size*sizeof(double));
//    double* hess = (double*)malloc(Y_size*sizeof(double));
//
//
//    LambdaRankObjective(Y, F, 1.0, Y_size, group, group_size, grad, hess);
//
//    int grad_nonzero = 0;
//    int grad_zero = 0;
//    for (int i = 0; i < Y_size; ++i) {
//        if(grad[i] == 0) grad_zero++;
//        else grad_nonzero++;
//    }
//
//    printf("grad_zero:%d\n", grad_zero);
//    printf("grad_nonzero:%d\n", grad_nonzero);
//
//    int hess_nonzero = 0;
//    int hess_zero = 0;
//    for (int i = 0; i < Y_size; ++i) {
//        if(hess[i] == 0) hess_zero++;
//        else hess_nonzero++;
//    }
//
//    printf("hess_zero:%d\n", hess_zero);
//    printf("hess_nonzero:%d\n", hess_nonzero);
//
//
//    printf("GRAD: \n");
//    print_array(grad, Y_size);
//
//    printf("HESS: \n");
//    print_array(hess, Y_size);
//
//}
//
//int main() {
//
//    test_lambda();
//
//    double* y = (double*)malloc(4*sizeof(double));
//    for (int i = 0; i < 5; ++i) {
//        y[i] = i;
//    }
//
//    double ndcg_res = calculate_ndcg(y,y,5);
//
//    printf("NDCG: %f\n", ndcg_res);
////
////    struct GradPair p;
////    p.grad = y;
////    p.hess = y;
//
//    //printf("aaa");
//}