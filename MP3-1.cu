// Heat Transfer Simulation
// MP3, Spring 2016, GPU Programming @ Auburn University
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>

#define N 131072            // Number of points in the rod
#define INITIAL_LEFT  1000.0  // Initial temperature at left end
#define INITIAL_RIGHT 0.0     // Initial temperature at right end
#define ALPHA 0.5             // Constant
#define MAX_TIMESTEPS 10000  // Maximum number of time steps

#define THREADS_PER_BLOCK 500

#define CUDA_CHECK(e) { \
cudaError_t err = (e); \
if (err != cudaSuccess) \
{ \
	fprintf(stderr, "CUDA error: %s, line %d, %s: %s\n", \
		__FILE__, __LINE__, #e, cudaGetErrorString(err)); \
	exit(EXIT_FAILURE); \
} \
}




static void check_result(double *result);
__global__ void heat_t(double *d_new_t, double *d_old_t,double *d_temp);

int main() {

    double *h_new_t = (double *)malloc(N * sizeof(double));
    double *h_old_t = (double *)malloc(N * sizeof(double));
	//double *h_tmp = (double *)malloc(N * sizeof(double));

    h_old_t[0] = INITIAL_LEFT;
    for (int i = 1; i < N-1; i++) {
        h_old_t[i] = 0.0;
    	}
    h_old_t[N-1] = INITIAL_RIGHT;


	double *d_new_t;
	CUDA_CHECK(cudaMalloc((void **)&d_new_t,N * sizeof(double))); //Memory allocation for new_t on device

	double *d_old_t;
	CUDA_CHECK(cudaMalloc((void **)&d_old_t,N * sizeof(double))); ////Memory allocation for old_t on device

    double *d_temp;
    CUDA_CHECK(cudaMalloc((void **)&d_temp,N * sizeof(double))); // Memory allocation for tmp variable - used for pointer swapping 


    //CUDA_CHECK(cudaMemcpy(d_new_t, h_new_t, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_old_t, h_old_t, N * sizeof(double), cudaMemcpyHostToDevice));

   double start = omp_get_wtime();

	/* Launch the CUDA kernel */

for (int x = 0; x < MAX_TIMESTEPS; x++){

	heat_t<<<128,1024>>>(d_new_t,d_old_t,d_temp);
 	    d_temp  = d_old_t;
        d_old_t = d_new_t;
        d_new_t = d_temp;


}
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
CUDA_CHECK(cudaMemcpy(h_old_t, d_old_t, N * sizeof(double), cudaMemcpyDeviceToHost));


    double stop = omp_get_wtime();

    // Show output (final temperatures)
    printf("Stopped after %d time steps\n", MAX_TIMESTEPS);
    printf("Simulation took %f seconds\n", stop - start);
    
    check_result(h_old_t);
    return 0;
}


__global__ void heat_t(double *d_new_t,double *d_old_t,double *d_temp)
{

int i;
i = threadIdx.x + blockIdx.x * gridDim.x ; 

if (i != N-1)
{
 d_new_t[0] = d_old_t[0];
    	
            d_new_t[i] = d_old_t[i] + ALPHA*(d_old_t[i-1] + d_old_t[i+1] - 2*d_old_t[i]);
}
else 
{
d_new_t[N-1] = d_old_t[N-1];
}

}

static void check_result(double *result) {
    char output[1024] = { 0 };
    char *out = output;
    
    // Display some of the computed results
    for (int i = 0; i < 6; i++) {
        out += sprintf(out, "%3.3f ", result[i]);
    }
    out += sprintf(out, "... ");
    for (int i = N-6; i < N; i++) {
        out += sprintf(out, "%3.3f ", result[i]);
    }
    printf("Computed: %s\n", output);

    // Display the expected output
    const char *expected = "1000.000 992.021 984.044 976.067 968.095 960.123 ... 0.000 0.000 0.000 0.000 0.000 0.000 ";
    printf("Expected: %s\n", expected);

    // Exit with a nonzero exit code if the two do not match
    assert(strcmp(output, expected) == 0);
}