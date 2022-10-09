/*
 *  * How to compile (assume cuda is installed at /usr/local/cuda-10.1/)
 *   *   nvcc -c -I/usr/local/cuda-10.1/include gesvdj_example.cpp 
 *    *   g++ -o gesvdj_example gesvdj_example.o -L/usr/local/cuda-10.1/lib64 -lcudart -lcusolver
 *     */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %20.16E\n", name, row+1, col+1, Areg);
        }
    }
}

int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    gesvdjInfo_t gesvdj_params = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    const int m = 3;
    const int n = 2;
    const int lda = m;
/*       | 1 2  |
 *        *   A = | 4 5  |
 *         *       | 2 1  |
 *          */
    double A[lda*n] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
    double U[lda*m]; /* m-by-m unitary matrix, left singular vectors  */
    double V[lda*n]; /* n-by-n unitary matrix, right singular vectors */
    double S[n];     /* numerical singular value */
/* exact singular values */
    double S_exact[n] = {7.065283497082729, 1.040081297712078};
    double *d_A = NULL;  /* device copy of A */
    double *d_S = NULL;  /* singular values */
    double *d_U = NULL;  /* left singular vectors */
    double *d_V = NULL;  /* right singular vectors */
    int *d_info = NULL;  /* error info */
    int lwork = 0;       /* size of workspace */
    double *d_work = NULL; /* devie workspace for gesvdj */
    int info = 0;        /* host copy of error info */

/* configuration of gesvdj  */
    const double tol = 1.e-7;
    const int max_sweeps = 15;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    const int econ = 0 ; /* econ = 1 for economy size */

/* numerical results of gesvdj  */
    double residual = 0;
    int executed_sweeps = 0;

    printf("example of gesvdj \n");
    printf("tol = %E, default value is machine zero \n", tol);
    printf("max. sweeps = %d, default value is 100\n", max_sweeps);
    printf("econ = %d \n", econ);

    printf("A = (matlab base-1)\n");
    printMatrix(m, n, A, lda, "A");
    printf("=====\n");

/* step 1: create cusolver handle, bind a stream */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 2: configuration of gesvdj */
    status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of tolerance is machine zero */
    status = cusolverDnXgesvdjSetTolerance(
        gesvdj_params,
        tol);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of max. sweeps is 100 */
    status = cusolverDnXgesvdjSetMaxSweeps(
        gesvdj_params,
        max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 3: copy A and B to device */
    cudaStat1 = cudaMalloc ((void**)&d_A   , sizeof(double)*lda*n);
    cudaStat2 = cudaMalloc ((void**)&d_S   , sizeof(double)*n);
    cudaStat3 = cudaMalloc ((void**)&d_U   , sizeof(double)*lda*m);
    cudaStat4 = cudaMalloc ((void**)&d_V   , sizeof(double)*lda*n);
    cudaStat5 = cudaMalloc ((void**)&d_info, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double)*lda*n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);

/* step 4: query workspace of SVD */
    status = cusolverDnDgesvdj_bufferSize(
        cusolverH,
        jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
              /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ, /* econ = 1 for economy size */
        m,    /* nubmer of rows of A, 0 <= m */
        n,    /* number of columns of A, 0 <= n  */
        d_A,  /* m-by-n */
        lda,  /* leading dimension of A */
        d_S,  /* min(m,n) */
              /* the singular values in descending order */
        d_U,  /* m-by-m if econ = 0 */
              /* m-by-min(m,n) if econ = 1 */
        lda,  /* leading dimension of U, ldu >= max(1,m) */
        d_V,  /* n-by-n if econ = 0  */
              /* n-by-min(m,n) if econ = 1  */
        lda,  /* leading dimension of V, ldv >= max(1,n) */
        &lwork,
        gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaMalloc((void**)&d_work , sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

/* step 5: compute SVD */
    status = cusolverDnDgesvdj(
        cusolverH,
        jobz,  /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
               /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ,  /* econ = 1 for economy size */
        m,     /* nubmer of rows of A, 0 <= m */
        n,     /* number of columns of A, 0 <= n  */
        d_A,   /* m-by-n */
        lda,   /* leading dimension of A */
        d_S,   /* min(m,n)  */
               /* the singular values in descending order */
        d_U,   /* m-by-m if econ = 0 */
               /* m-by-min(m,n) if econ = 1 */
        lda,   /* leading dimension of U, ldu >= max(1,m) */
        d_V,   /* n-by-n if econ = 0  */
               /* n-by-min(m,n) if econ = 1  */
        lda,   /* leading dimension of V, ldv >= max(1,n) */
        d_work,
        lwork,
        d_info,
        gesvdj_params);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(U, d_U, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(V, d_V, sizeof(double)*lda*n, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(S, d_S, sizeof(double)*n    , cudaMemcpyDeviceToHost);
    cudaStat4 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    cudaStat5 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);

    if ( 0 == info ){
        printf("gesvdj converges \n");
    }else if ( 0 > info ){
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }else{
        printf("WARNING: info = %d : gesvdj does not converge \n", info );
    }

    printf("S = singular values (matlab base-1)\n");
    printMatrix(n, 1, S, lda, "S");
    printf("=====\n");

    printf("U = left singular vectors (matlab base-1)\n");
    printMatrix(m, m, U, lda, "U");
    printf("=====\n");

    printf("V = right singular vectors (matlab base-1)\n");
    printMatrix(n, n, V, lda, "V");
    printf("=====\n");

/* step 6: measure error of singular value */
    double ds_sup = 0;
    for(int j = 0; j < n; j++){
        double err = fabs( S[j] - S_exact[j] );
        ds_sup = (ds_sup > err)? ds_sup : err;
    }
    printf("|S - S_exact|_sup = %E \n", ds_sup);

    status = cusolverDnXgesvdjGetSweeps(
        cusolverH,
        gesvdj_params,
        &executed_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    status = cusolverDnXgesvdjGetResidual(
        cusolverH,
        gesvdj_params,
        &residual);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    printf("residual |A - U*S*V**H|_F = %E \n", residual );
    printf("number of executed sweeps = %d \n", executed_sweeps );

/*  free resources  */
    if (d_A    ) cudaFree(d_A);
    if (d_S    ) cudaFree(d_S);
    if (d_U    ) cudaFree(d_U);
    if (d_V    ) cudaFree(d_V);
    if (d_info) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);
    if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);

    cudaDeviceReset();
    return 0;
}
