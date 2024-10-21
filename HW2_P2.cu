#include <stdio.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}
#define TILE_WIDTH 32
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

__global__ void matrix_multiplication_kernel1(float* A, float* B, float* C, int m, int n, int k)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    float C_rc = 0;
    if (r < m && c < k) {
        for (int i = 0; i < n; ++i) {
            C_rc += A[r * n + i] * B[i * k + c];
        }
    }
    C[r * k + c] = C_rc;
}

__global__ void matrix_multiplication_kernel2(float* A, float* B, float* C, int m, int n, int k)
{
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    float C_rc = 0;

    int r_A, c_A, r_B, c_B;
    for (int tile = 0; tile < (n - 1) / TILE_WIDTH + 1; ++tile) {
        r_A = r; c_A = tile * TILE_WIDTH + threadIdx.x;
        r_B = tile * TILE_WIDTH + threadIdx.y; c_B = c;
        s_A[threadIdx.y][threadIdx.x] = (r_A < m && c_A < n) ? A[r_A * n + c_A] : 0;
        s_B[threadIdx.y][threadIdx.x] = (r_B < n && c_B < k) ? B[r_B * k + c_B] : 0;
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            C_rc += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        }
        __syncthreads();
    }

    C[r * k + c] = C_rc;
}

void matrix_multiplication(float* A, float* B, float* C, int m, int n, int k,
    bool useDevice = false, dim3 blockSize = dim3(1),int kernelType=1)
{
    GpuTimer timer;
    timer.Start();
    if (useDevice == false)
    {
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < k; ++c) {
                C[r * k + c] = 0;
                for (int i = 0; i < n; ++i) {
                    C[r * k + c] += A[r * n + i] * B[i * k + c];
                }
            }
        }
    }
    else // Use device
    {
        // TODO: Allocate device memories
        float* d_A, * d_B, * d_C;
        CHECK(cudaMalloc((void**) &d_A, m * n * sizeof(float)));
        CHECK(cudaMalloc((void**) &d_B, n * k * sizeof(float)));
        CHECK(cudaMalloc((void**) &d_C, m * k * sizeof(float)));

        // TODO: Copy data to device memories
        CHECK(cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice));
        
        dim3 gridSize((m - 1) / blockSize.x + 1, (k - 1) / blockSize.y + 1); // TODO: Compute gridSize
        
		if (kernelType == 1)
			matrix_multiplication_kernel1<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);
		else if (kernelType == 2)
			matrix_multiplication_kernel2<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);

        // TODO: Copy result from device memory
        CHECK(cudaMemcpy(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost));

        // TODO: Free device memories
        CHECK(cudaFree(d_A));
        CHECK(cudaFree(d_B));
        CHECK(cudaFree(d_C));
		
		printf("Grid size: %d * %d, block size: %d * %d\n", 
			gridSize.x,gridSize.y, blockSize.x,blockSize.y);

    }
    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (%s): %f ms\n",
        useDevice == true ? "use device" : "use host", time);
}

float checkCorrectness(float * a1, float* a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)	
		err += abs(a1[i] - a2[i]);
	err /= n;
	return err;
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("****************************\n\n");

}
int main(int argc, char** argv)
{
	printDeviceInfo();
	
	//Declare variables
    float* h_A; // The A matrix
    float* h_B; // The B matrix
    float* h_C; // The output C matrix
    float* correct_C; // The output C matrix

    int m;    // number of rows in the matrix A
    int n; // number of columns in the matrix A, number of rows in the matrix B
    int k; // number of columns in the matrix B

    m = (1 << 10);
    n = (1 << 9);
    k = (1 << 10);

    // Set up input data
    h_A = (float*)malloc(m * n * sizeof(float));
    h_B = (float*)malloc(n * k * sizeof(float));
    h_C = (float*)malloc(m * k * sizeof(float));
    correct_C = (float*)malloc(m * k * sizeof(float));

    for (int i = 0; i < m; i++)
        for (int j = 0;j < n;j++)
            h_A[i*n+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
 
    for (int i = 0; i < n; i++)
        for (int j = 0;j < k;j++)
            h_B[i*k+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);


    // Add vectors (on host)
    matrix_multiplication(h_A,h_B,correct_C,m,n,k);
	printf("\n");

	dim3 blockSize(32, 32); // Default
	if (argc == 3)
	{
		blockSize.x = atoi(argv[1]);
		blockSize.y = atoi(argv[2]);
	} 
    // Add in1 & in2 on device
	printf("Basic Matrix Multiplication:\n");
    matrix_multiplication(h_A, h_B, h_C, m, n, k, true,blockSize,1);
	float err = checkCorrectness(h_C, correct_C,m*k);
	printf("Error between device result and host result: %f\n\n", err);

	printf("Shared memory Matrix Multiplication:\n");
    matrix_multiplication(h_A, h_B, h_C, m, n, k, true,blockSize,2);
	err = checkCorrectness(h_C, correct_C,m*k);
	printf("Error between device result and host result: %f", err);	
	
    free(h_A);
    free(h_B);
    free(h_C);
    free(correct_C);

    return 0;
}
