#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "arr_utils.h"

double calculate_dcg(double *y, int size)
{
    double dcg = 0.0;
    for(int i =0; i< size; ++i)
    {
        dcg += (pow2((int)y[i]) - 1)/log(2.0 + i);
    }
    return dcg;
}

double calculate_dcg_by_idxs(double *y, int* idxs, int size)
{
    double dcg = 0.0;
    for(int i =0; i< size; ++i)
    {
        dcg += (pow2((int)y[idxs[i]]) - 1)/log(2.0 + i);
    }
    return dcg;
}

double divide_double(double numerator, double denominator)
{
    if(numerator == 0) return 0.0;
    else
    {
        if(denominator == 0) return 0.0;
        else return numerator/denominator;
    }
}



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

void calculate_pairwise_delta_ndcg(double* y, double* f, int size, double** res)
{
    int* f_sorted_idxs = argsort(f, size);
    int* y_sorted_idxs = argsort(y, size);

    int* inv_f_idxs = (int*)malloc(size* sizeof(int));
    for (int i = 0; i < size; ++i) {
        inv_f_idxs[f_sorted_idxs[i]] = i;
    }

    double ideal_dcg = calculate_dcg_by_idxs(y, y_sorted_idxs, size);

    int ndcg_at_5_size = (int)fmin(5, size);


    double* dcg_arr = (double*)malloc(size* sizeof(double));
    double* log_arr = (double*)malloc(size* sizeof(double));
    double* rel_arr = (double*)malloc(size* sizeof(double));
    double dcg_sum = 0;
    for (int i = 0; i < size; ++i) {
        log_arr[i] = log(2.0 + i);
        rel_arr[i] = (pow2((int)y[f_sorted_idxs[i]]) - 1);
        dcg_arr[i] = rel_arr[i]/log_arr[i];

        dcg_sum += dcg_arr[i];
    }

//    double dcg_1 = calculate_dcg_by_idxs(y, f_sorted_idxs, size);
    double ndcg_1 = divide_double(dcg_sum, ideal_dcg);


    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {

            int d1_sorted_idx = inv_f_idxs[i];
            int d2_sorted_idx = inv_f_idxs[j];

            swap_arr_int(f_sorted_idxs, d1_sorted_idx, d2_sorted_idx);

            double dcg_2 = calculate_dcg_by_idxs(y, f_sorted_idxs, ndcg_at_5_size);
            double ndcg_2 = divide_double(dcg_2, ideal_dcg);

            res[i][j] = ndcg_1 - ndcg_2;

            swap_arr_int(f_sorted_idxs, d2_sorted_idx, d1_sorted_idx);


            // swap i dobument with j document in sorting by F and recompute ndcg
            double NEW_dcg_2 = dcg_sum;
            double rel_d1 = rel_arr[d1_sorted_idx];//dcg_arr[d1_sorted_idx] * log_arr[d1_sorted_idx];
            double rel_d2 = rel_arr[d2_sorted_idx];//dcg_arr[d1_sorted_idx] * log_arr[d2_sorted_idx];

            dcg_2 = dcg_2 - dcg_arr[d1_sorted_idx] - dcg_arr[d2_sorted_idx];

            dcg_2 += rel_d1/log_arr[d1_sorted_idx] + rel_d2/log_arr[d2_sorted_idx];
            double NEW_ndcg_2 =0;
            NEW_ndcg_2 = divide_double(dcg_2, ideal_dcg);

            if(fabs(NEW_ndcg_2 - ndcg_2) > 1e-10)
            {
                printf("TRUE: %lf\n",ndcg_2);
                printf("NEW: %lf\n",NEW_ndcg_2);
            }
            //res[i][j] = ndcg_1 - ndcg_2;
            //res[i][j] = ndcg_1 - ndcg_2;

        }
    }

    free(f_sorted_idxs);
    free(y_sorted_idxs);
    free(inv_f_idxs);
    free(dcg_arr);
    free(log_arr);
    free(rel_arr);
}



void pairwise_diff(double* x, int size, double** res)
{
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            res[i][j] = x[i] - x[j];
        }
    }
}





extern void LambdaRankObjective(double* Y,
                                    double* F,
                                    double sigma,
                                    int size,
                                    int* group, int group_size,
                                    double* grad, double* hess)
{

//    double _time_delta_sum = 0.0;
//    double _time_delta_max = 0.0;
//    int _time_delta_counter = 0;


    int group_start = 0;
    int group_len =  0;

//    int count_grad_zero = 0;
//    int count_grad_nonzero = 0;

    int max_group_len = 0;
    for(int i =0; i<group_size; ++i) {
        if(group[i] > max_group_len)
            max_group_len = group[i];
    }

    double** pairwise_s = (double**)malloc(max_group_len*sizeof(double*));
    double** pairwise_f = (double**)malloc(max_group_len*sizeof(double*));
    double** pairwise_delta_ndcg = (double**)malloc(max_group_len*sizeof(double*));
    for (int i = 0; i < max_group_len; ++i) {
        pairwise_s[i] = (double*)malloc(max_group_len*sizeof(double));
        pairwise_f[i] = (double*)malloc(max_group_len*sizeof(double));
        pairwise_delta_ndcg[i] = (double*)malloc(max_group_len*sizeof(double));
    }

    for (int group_i = 0; group_i < group_size; group_i++)
    {

        group_start += group_len;
        group_len = group[group_i];

        double* group_Y = Y + group_start;
        double* group_F = F + group_start;

        pairwise_diff(group_Y, group_len, pairwise_s);
        sign_matrix(pairwise_s, group_len, group_len);

        pairwise_diff(group_F, group_len, pairwise_f);

        //TEST PROFILING TIME
//        clock_t t;
//        t = clock();


        calculate_pairwise_delta_ndcg(group_Y, group_F, group_len, pairwise_delta_ndcg);

        // TEST PROFILING TIME
//        t = clock() - t;
//        double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
//        _time_delta_sum += time_taken;
//        _time_delta_counter++;
//        if(time_taken > _time_delta_max)
//            _time_delta_max = time_taken;

        for (int i = 0; i < group_len; ++i)
        {
            double cur_grad = 0;
            double cur_hess = 0;

            for (int j = 0; j < group_len; ++j) {
                //double _delta_ndcg = delta_ndcg(group_Y, group_F,  i, j, group_len);
                double delta_ndcg = pairwise_delta_ndcg[i][j];
                double pij = sigm( sigma * fabs(pairwise_f[i][j]) );
                cur_grad += pairwise_s[i][j] *(-sigma * delta_ndcg * pij);
                cur_hess += fabs(pairwise_s[i][j]) * sigma * sigma * delta_ndcg * pij * (1.0 - pij);
            }

//            if(cur_grad == 0)
//                count_grad_zero++;
//            else
//                count_grad_nonzero++;

            grad[group_start + i] = cur_grad;
            hess[group_start + i] = cur_hess;
        }
    }
    free_mat(pairwise_s, max_group_len);
    free_mat(pairwise_f, max_group_len);
    free_mat(pairwise_delta_ndcg, max_group_len);

    //TEST PROFILING TIME
//
//    printf("calculate_pairwise_delta_ndcg took %f seconds to execute (mean) \n", _time_delta_sum/_time_delta_counter);
//    printf("calculate_pairwise_delta_ndcg took %f seconds to execute (max) \n", _time_delta_max);
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

void test_lambda()
{
    int Y_size = count_lines("Y_test.txt");
    int F_size = count_lines("F_test.txt");
    int group_size = count_lines("group_test.txt");

    double* Y = (double*)malloc(Y_size*sizeof(double));
    double* F = (double*)malloc(F_size*sizeof(double));
    int* group = (int*)malloc(group_size*sizeof(int));

    read_file(Y, "Y_test.txt");
    read_file(F, "F_test.txt");
    read_file_int(group, "group_test.txt");

    double* grad = (double*)malloc(Y_size*sizeof(double));
    double* hess = (double*)malloc(Y_size*sizeof(double));



    clock_t t;
    t = clock();


    LambdaRankObjective(Y, F, 1.0, Y_size, group, group_size, grad, hess);

    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
    printf("LambdaRankObjective took %f seconds to execute \n", time_taken);



    int grad_nonzero = 0;
    int grad_zero = 0;
    for (int i = 0; i < Y_size; ++i) {
        if(grad[i] == 0) grad_zero++;
        else grad_nonzero++;
    }

    printf("grad_zero:%d\n", grad_zero);
    printf("grad_nonzero:%d\n", grad_nonzero);

    int hess_nonzero = 0;
    int hess_zero = 0;
    for (int i = 0; i < Y_size; ++i) {
        if(hess[i] == 0) hess_zero++;
        else hess_nonzero++;
    }

    printf("hess_zero:%d\n", hess_zero);
    printf("hess_nonzero:%d\n", hess_nonzero);

//
//    printf("GRAD: \n");
//    print_array(grad, Y_size);
//
//    printf("HESS: \n");
//    print_array(hess, Y_size);

}


void test_arr_func()
{
    int size = 4;
    double arr[] = {2.0, 1.0, 3.0, 5.0 };
    double sorted[] = {5.0, 3.0, 2.0, 1.0};
    int sorted_idxs[] = {3, 2, 0, 1};

    printf("ARR:\n");
    print_array(arr, size);

    printf("SORTED_ARR:\n");
    print_array(sorted, size);

    printf("SORTED_IDXS:\n");
    print_array_int(sorted_idxs, size);



    double* arr_sort = copy_arr(arr, size);
    sort_arr(arr_sort, size);

    for (int i = 0; i < size; i++) {
        assert(sorted[i] == arr_sort[i]);
    }

    int* idxs = argsort(arr, size);
    for (int i = 0; i < size; ++i) {
        assert(idxs[i] ==  sorted_idxs[i]);
    }


    printf("MY_SORTED_ARR:\n");
    print_array(arr_sort, size);

    printf("MY_SORTED_IDXS:\n");
    print_array_int(sorted_idxs, size);
}

int main() {

    //test_arr_func();
    test_lambda();

//    double* y = (double*)malloc(4*sizeof(double));
//    for (int i = 0; i < 5; ++i) {
//        y[i] = i;
//    }
//
//    double ndcg_res = calculate_ndcg(y,y,5);
//
//    printf("NDCG: %f\n", ndcg_res);
//
//    struct GradPair p;
//    p.grad = y;
//    p.hess = y;

    //printf("aaa");
}