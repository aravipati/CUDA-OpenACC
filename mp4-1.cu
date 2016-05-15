// Brute-force Key Search - TEA Encryption with 31-bit Key
// MP4, Spring 2016, GPU Programming @ Auburn University
#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#define CUDA_CHECK(e) { \
cudaError_t err = (e); \
if (err != cudaSuccess) \
{ \
    fprintf(stderr, "CUDA error: %s, line %d, %s: %s\n", \
        __FILE__, __LINE__, #e, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
} \
}
void encrypt_cpu(uint32_t *data, const uint32_t *key) {
    uint32_t v0=data[0], v1=data[1], sum=0, i;             /* set up */
    uint32_t delta=0x9e3779b9;                             /* a key schedule constant */
    uint32_t k0=key[0], k1=key[1], k2=key[2], k3=key[3];   /* cache key */
    for (i=0; i < 32; i++) {                               /* basic cycle start */
        sum += delta;
        v0 += ((v1<<4) + k0) ^ (v1 + sum) ^ ((v1>>5) + k1);
        v1 += ((v0<<4) + k2) ^ (v0 + sum) ^ ((v0>>5) + k3);
    }                                                      /* end cycle */
    data[0]=v0; data[1]=v1;
}


__global__ void encrypt_gpu(uint32_t *data,  uint32_t *key, uint32_t * h_key_result);
/* Data to test with (this should be easy to change) */
const uint32_t orig_data[2] = { 0xDEADBEEF, 0x0BADF00D };
const uint32_t encrypted[2] = { 0xFF305F9B, 0xB9BDCECE  };

__global__ void encrypt_gpu(uint32_t *data,  uint32_t *key,uint32_t * h_key_result) {
const uint32_t encrypted[2] = { 0xFF305F9B, 0xB9BDCECE};
    
int x = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t v0=data[0], v1=data[1], sum=0, i;             /* set up */
    uint32_t delta=0x9e3779b9;                             /* a key schedule constant */
    uint32_t k0=key[0] * 32 * 16 + x; 
    uint32_t k1=key[1] * 32 * 16 + x; 
    uint32_t k2=key[2] * 32 * 16 + x; 
    uint32_t k3=key[3] * 32 * 16 + x; 

    for (i=0; i < 128; i++) {                               /* basic cycle start */
        sum += delta;
        v0 += ((v1<<4) + k0) ^ (v1 + sum) ^ ((v1>>5) + k1);
        v1 += ((v0<<4) + k2) ^ (v0 + sum) ^ ((v0>>5) + k3);
    }
     if (v0 == encrypted[0] && v1 == encrypted[1]) {
            data[0]=v0; data[1]=v1;
            
            h_key_result[0]=k0;
            h_key_result[1]=k1;
            h_key_result[2]=k2;
            h_key_result[3]=k3;
            
            
        }                                                     
}



int main() {
    uint32_t key[4]  = { 1, 1, 1 ,1};
    uint32_t h_key_result[4] = { 1, 1, 1 ,1};
    uint32_t data[2] = { 0, 0 };
    uint32_t *dev_key_result;
    uint32_t *dev_key;
    uint32_t *dev_data;

    printf("Starting (this may take a while)...\n");
    double start = omp_get_wtime();
    /* Try every possible 28-bit integer... */

CUDA_CHECK(cudaMalloc((void**)&dev_key, 4* sizeof(uint32_t))); 
CUDA_CHECK(cudaMalloc((void**)&dev_data, 2* sizeof(uint32_t))); 
CUDA_CHECK(cudaMalloc((void**)&dev_key_result, 4* sizeof(uint32_t))); 



    for (uint32_t x = 0; x < 28; x++) {
        /* Try encrypting the data with the key { x, x, x, x } */
        data[0] = orig_data[0];
        data[1] = orig_data[1];
        key[0] = key[1] = key[2] = key[3] = x;
CUDA_CHECK(cudaMemcpy(dev_key, key, 4*sizeof(uint32_t), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(dev_data, data, 2*sizeof(uint32_t), cudaMemcpyHostToDevice));
        encrypt_gpu<<<32,16>>>(dev_data, dev_key,dev_key_result);
    }

CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
CUDA_CHECK(cudaMemcpy(data, dev_data,2*sizeof(uint32_t), cudaMemcpyDeviceToHost));
CUDA_CHECK(cudaMemcpy(key, dev_key,4*sizeof(uint32_t), cudaMemcpyDeviceToHost));
CUDA_CHECK(cudaMemcpy(h_key_result, dev_key_result,4*sizeof(uint32_t), cudaMemcpyDeviceToHost));


    printf("Elapsed time: %f seconds\n", omp_get_wtime() - start);

    /* Assume the above loop will find a key */

    printf("Found key: (hexadecimal) %08x %08x %08x %08x\n", h_key_result[0], h_key_result[1], h_key_result[2], h_key_result[3]);
    data[0] = orig_data[0];
    data[1] = orig_data[1];
    printf("The original values are (hexadecimal):  %08x %08x\n", data[0], data[1]);
    encrypt_cpu(data, h_key_result);
    printf("The encrypted values are (hexadecimal): %08x %08x\n", data[0], data[1]);
    printf("They should be:                         %08x %08x\n", encrypted[0], encrypted[1]);
    if (data[0] == encrypted[0] && data[1] == encrypted[1]) {
        printf("SUCCESS!\n");
        return 0;
    } else {
        printf("FAILED\n");
        return 1;
    }
    
    

    



}