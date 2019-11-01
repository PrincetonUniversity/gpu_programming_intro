/* CPU VERSION */

// modified from https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
 
void vecAdd(double *a, double *b, double *c, int n)
{
    int i;
    for(i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = 2000;
 
    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate memory for each vector on host
    fprintf(stderr, "Allocating memory and populating arrays of length %d ...", n);
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
 
    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);
    }

    fprintf(stderr, " done.\n");
    fprintf(stderr, "Performing vector addition (timer started) ...");
    StartTimer();

    // add the two vectors
    vecAdd(h_a, h_b, h_c, n);
 
    double runtime = GetTimer();
    fprintf(stderr, " done in %.2f s.\n", runtime / 1000);
 
    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    double tol = 1e-6;
    if (fabs(sum/n - 1.0) > tol) printf("Warning: potential numerical problems.\n"); 
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
