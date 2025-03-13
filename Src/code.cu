% % writefile matrix_multiplication.cu

#include <stdio.h>
#include <stdlib.h>

    // Function to initialize a matrix with random values
    void
    initMatrix(float *matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        matrix[i] = (float)(rand() % 100);
    }
}

// CPU matrix multiplication
void matrixMulCPU(float *A, float *B, float *C, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < n; k++)
            {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// GPU matrix multiplication - 2D kernel
__global__ void matrixMulGPU_2D(float *A, float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (row < n && col < n)
    {
        float sum = 0.0;
        for (int k = 0; k < n; k++)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// GPU matrix multiplication - 1D kernel
__global__ void matrixMulGPU_1D(float *A, float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Linear thread index
    int row = idx / n;                               // Calculate row from index
    int col = idx % n;                               // Calculate column from index

    if (row < n && col < n)
    {
        float sum = 0.0;
        for (int k = 0; k < n; k++)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Function to compare CPU and GPU results
void verifyResults(float *C_CPU, float *C_GPU, int n)
{
    for (int i = 0; i < n * n; i++)
    {
        if (fabs(C_CPU[i] - C_GPU[i]) > 1e-4)
        {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, C_CPU[i], C_GPU[i]);
            return;
        }
    }
    printf("Results are correct!\n");
}

int main()
{
    srand(time(NULL));

    int sizes[] = {256, 512}; // Different matrix sizes to test
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int test = 0; test < num_sizes; test++)
    {
        int N = sizes[test];
        printf("\nMatrix Size: %d x %d\n", N, N);

        int size = N * N * sizeof(float);

        // Allocate host memory
        float *h_A = (float *)malloc(size);
        float *h_B = (float *)malloc(size);
        float *h_C_CPU = (float *)malloc(size);
        float *h_C1_GPU = (float *)malloc(size);
        float *h_C2_GPU = (float *)malloc(size);

        // Initialize matrices
        initMatrix(h_A, N * N);
        initMatrix(h_B, N * N);

        // CUDA Events for CPU Timing
        cudaEvent_t startCPU, endCPU;
        cudaEventCreate(&startCPU);
        cudaEventCreate(&endCPU);

        cudaEventRecord(startCPU);
        matrixMulCPU(h_A, h_B, h_C_CPU, N);
        cudaEventRecord(endCPU);
        cudaEventSynchronize(endCPU);
        cudaDeviceSynchronize();

        float cpu_time;
        cudaEventElapsedTime(&cpu_time, startCPU, endCPU);
        printf("CPU Execution Time: %f ms\n", cpu_time);
        printf("\n");

        // Allocate device memory
        float *d_A, *d_B, *d_C1, *d_C2;
        cudaMalloc((void **)&d_A, size);
        cudaMalloc((void **)&d_B, size);
        cudaMalloc((void **)&d_C1, size);
        cudaMalloc((void **)&d_C2, size);

        // Data transfer and kernel timing for 1D kernel
        cudaEvent_t startHtoD1D, endHtoD1D, startKernel1D, endKernel1D, startDtoH1D, endDtoH1D;
        cudaEventCreate(&startHtoD1D);
        cudaEventCreate(&endHtoD1D);
        cudaEventCreate(&startKernel1D);
        cudaEventCreate(&endKernel1D);
        cudaEventCreate(&startDtoH1D);
        cudaEventCreate(&endDtoH1D);

        // Host to Device Transfer (1D Kernel)
        cudaEventRecord(startHtoD1D);
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        cudaEventRecord(endHtoD1D);
        cudaEventSynchronize(endHtoD1D);
        cudaDeviceSynchronize();

        float htoD_time_1D;
        cudaEventElapsedTime(&htoD_time_1D, startHtoD1D, endHtoD1D);

        // 1D Kernel Execution
        int threads_per_block_1D = 256;
        int num_blocks_1D = (N * N + threads_per_block_1D - 1) / threads_per_block_1D;

        cudaEventRecord(startKernel1D);
        matrixMulGPU_1D<<<num_blocks_1D, threads_per_block_1D>>>(d_A, d_B, d_C1, N);
        cudaEventRecord(endKernel1D);
        cudaEventSynchronize(endKernel1D);
        cudaDeviceSynchronize();

        float kernel_time_1D;
        cudaEventElapsedTime(&kernel_time_1D, startKernel1D, endKernel1D);

        // Device to Host Transfer (1D Kernel)
        cudaEventRecord(startDtoH1D);
        cudaMemcpy(h_C1_GPU, d_C1, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(endDtoH1D);
        cudaEventSynchronize(endDtoH1D);
        cudaDeviceSynchronize();

        float dtoH_time_1D;
        cudaEventElapsedTime(&dtoH_time_1D, startDtoH1D, endDtoH1D);

        float total_transfer_time_1D = htoD_time_1D + dtoH_time_1D;
        float total_time_1D = total_transfer_time_1D + kernel_time_1D;
        printf("GPU Execution Time (1D Kernel): %f ms\n", kernel_time_1D);
        printf("GPU Data Transfer Time (HtoD, 1D Kernel): %f ms\n", htoD_time_1D);
        printf("GPU Data Transfer Time (DtoH, 1D Kernel): %f ms\n", dtoH_time_1D);
        printf("Total Data Transfer Time (1D Kernel): %f ms\n", total_transfer_time_1D);
        printf("Total Time (1D Kernel): %f ms\n", total_time_1D);
        verifyResults(h_C_CPU, h_C1_GPU, N);
        printf("\n");

        // Data transfer and kernel timing for 2D kernel
        cudaEvent_t startHtoD2D, endHtoD2D, startKernel2D, endKernel2D, startDtoH2D, endDtoH2D;
        cudaEventCreate(&startHtoD2D);
        cudaEventCreate(&endHtoD2D);
        cudaEventCreate(&startKernel2D);
        cudaEventCreate(&endKernel2D);
        cudaEventCreate(&startDtoH2D);
        cudaEventCreate(&endDtoH2D);

        // Host to Device Transfer (2D Kernel)
        cudaEventRecord(startHtoD2D);
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        cudaEventRecord(endHtoD2D);
        cudaEventSynchronize(endHtoD2D);
        cudaDeviceSynchronize();

        float htoD_time_2D;
        cudaEventElapsedTime(&htoD_time_2D, startHtoD2D, endHtoD2D);

        // 2D Kernel Execution
        dim3 threads_per_block_2D(16, 16);
        dim3 num_blocks_2D((N + 16 - 1) / 16, (N + 16 - 1) / 16);

        cudaEventRecord(startKernel2D);
        matrixMulGPU_2D<<<num_blocks_2D, threads_per_block_2D>>>(d_A, d_B, d_C2, N);
        cudaEventRecord(endKernel2D);
        cudaEventSynchronize(endKernel2D);
        cudaDeviceSynchronize();

        float kernel_time_2D;
        cudaEventElapsedTime(&kernel_time_2D, startKernel2D, endKernel2D);

        // Device to Host Transfer (2D Kernel)
        cudaEventRecord(startDtoH2D);
        cudaMemcpy(h_C2_GPU, d_C2, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(endDtoH2D);
        cudaEventSynchronize(endDtoH2D);
        cudaDeviceSynchronize();

        float dtoH_time_2D;
        cudaEventElapsedTime(&dtoH_time_2D, startDtoH2D, endDtoH2D);

        float total_transfer_time_2D = htoD_time_2D + dtoH_time_2D;
        float total_time_2D = total_transfer_time_2D + kernel_time_2D;
        printf("GPU Execution Time (2D Kernel): %f ms\n", kernel_time_2D);
        printf("GPU Data Transfer Time (HtoD, 2D Kernel): %f ms\n", htoD_time_2D);
        printf("GPU Data Transfer Time (DtoH, 2D Kernel): %f ms\n", dtoH_time_2D);
        printf("Total Data Transfer Time (2D Kernel): %f ms\n", total_transfer_time_2D);
        printf("Total Time (2D Kernel): %f ms\n", total_time_2D);
        verifyResults(h_C_CPU, h_C2_GPU, N);
        printf("\n");

        // Cleanup
        free(h_A);
        free(h_B);
        free(h_C_CPU);
        free(h_C1_GPU);
        free(h_C2_GPU);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C1);
        cudaFree(d_C2);

        cudaEventDestroy(startCPU);
        cudaEventDestroy(endCPU);
        cudaEventDestroy(startHtoD1D);
        cudaEventDestroy(endHtoD1D);
        cudaEventDestroy(startKernel1D);
        cudaEventDestroy(endKernel1D);
        cudaEventDestroy(startDtoH1D);
        cudaEventDestroy(endDtoH1D);
        cudaEventDestroy(startHtoD2D);
        cudaEventDestroy(endHtoD2D);
        cudaEventDestroy(startKernel2D);
        cudaEventDestroy(endKernel2D);
        cudaEventDestroy(startDtoH2D);
        cudaEventDestroy(endDtoH2D);
    }

    return 0;
}
