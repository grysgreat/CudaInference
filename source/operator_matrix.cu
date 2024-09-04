#include "stdio.h"
#include "operator_matrix.cuh"
// dim3 block(32);
// dim3 grid(rows/32);
__global__ void scopy(float * matrix_in,float * matrix_out,int rows,int cols)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    for (int j = 0; j < cols; ++j) {
        matrix_out[tid*cols+j] = matrix_in[tid*cols+j];
    }
}





const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;


void strans(float *odata, float *idata,int rows,int cols){
    dim3 dimGrid(rows/TILE_DIM, cols/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);    

    strans_cuda<<<dimGrid,dimBlock>>>(odata,idata);
    cudaDeviceSynchronize();
}


float max_abs(float* d_array,float* d_work,float* c_work, int size){
    // calculating number of blocks based on array size
    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    max_abs_in_array<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_array, d_work, size);
    cudaDeviceSynchronize();

    cudaMemcpy(c_work, d_work, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost);
    // Finish the reduction on CPU
    float max_value = c_work[0];
    for(int i = 1; i < blocksPerGrid; i++) {
        if (c_work[i] > max_value) {
            max_value = c_work[i];
        }
    }
    return max_value;
}

float min_abs(float* d_array,float* d_work,float* c_work, int size){
    // calculating number of blocks based on array size
    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    min_abs_in_array<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_array, d_work, size);
    cudaDeviceSynchronize();

    cudaMemcpy(c_work, d_work, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost);
    // Finish the reduction on CPU
    float min_value = c_work[0];
    for(int i = 1; i < blocksPerGrid; i++) {
        if (c_work[i] < min_value) {
            min_value = c_work[i];
        }
    }
    return min_value;
}

float avg_abs(float* d_array,float* d_work,float* c_work, int size){
    // calculating number of blocks based on array size
    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    sum_abs_in_array<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_array, d_work, size);
    cudaDeviceSynchronize();

    cudaMemcpy(c_work, d_work, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost);
    // Finish the reduction on CPU
    float avg_abs = 0;
    double tmp;
    for(int i = 0; i < blocksPerGrid; i++) {
        tmp += c_work[i];
    }
    avg_abs= tmp/size;
    return avg_abs;
}


void max_abs_vec(float* d_array,float* d_work,float* c_work,float* output, int rows, int cols,char type){
    if(type == 'c'){
        float *d_array_copy;

        cudaMalloc((void**)&d_array_copy, sizeof(float) * rows*cols);

        strans(d_array_copy,d_array,rows,cols);

        max_abs_vec(d_array_copy,d_work,c_work,output, cols, rows,'r');
        
    } else {
    
        
        // calculating number of blocks based on array size
        int size = rows*cols;
        
        int threadsPerBlock = 1024;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        max_abs_in_array<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_array, d_work, size);
        cudaDeviceSynchronize();

        cudaMemcpy(c_work, d_work, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost);
        // Finish the reduction on CPU

        int len = cols/threadsPerBlock;
        
        for(int i = 0; i < rows; i++) {
            float max_value = 0;
            for(int j=0;j<len;j++){
                max_value = max(c_work[i*len+j],max_value);
            }
            output[i] = max_value;
        }
    }
    return ;
}


void min_abs_vec(float* d_array,float* d_work,float* c_work,float* output, int rows, int cols,char type){
    if(type == 'c'){
        float *d_array_copy;

        cudaMalloc((void**)&d_array_copy, sizeof(float) * rows*cols);

        strans(d_array_copy,d_array,rows,cols);

        min_abs_vec(d_array_copy,d_work,c_work,output, cols, rows,'r');
        
    } else {
    
        
        // calculating number of blocks based on array size
        int size = rows*cols;
        
        int threadsPerBlock = 1024;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        min_abs_in_array<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_array, d_work, size);
        cudaDeviceSynchronize();

        cudaMemcpy(c_work, d_work, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost);
        // Finish the reduction on CPU

        int len = cols/threadsPerBlock;
        
        for(int i = 0; i < rows; i++) {
            float min_value = c_work[i*len];
            for(int j=1;j<len;j++){
                min_value = min(c_work[i*len+j],min_value);
            }
            output[i] = min_value;
        }
    }
    return ;
}


void avg_abs_vec(float* d_array,float* d_work,float* c_work,float* output, int rows, int cols,char type){
    if(type == 'c'){
        float *d_array_copy;

        cudaMalloc((void**)&d_array_copy, sizeof(float) * rows*cols);

        strans(d_array_copy,d_array,rows,cols);

        avg_abs_vec(d_array_copy,d_work,c_work,output, cols, rows,'r');
        
    } else {
        // calculating number of blocks based on array size
        int size = rows*cols;
        
        int threadsPerBlock = 1024;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        sum_abs_in_array<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_array, d_work, size);
        cudaDeviceSynchronize();

        cudaMemcpy(c_work, d_work, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost);
        // Finish the reduction on CPU

        int len = cols/threadsPerBlock;
        
        for(int i = 0; i < rows; i++) {
            double sum_value = 0;
            for(int j=0;j<len;j++){
                sum_value += c_work[i*len+j];
            }
            output[i] = sum_value/((double)cols);
        }
    }
    return ;
}

void quantitize_int8(float * matrix_in,int8_t * matrix_out,int nx,int ny,float lambda){
    dim3 block(32, 32);
    //二维线程网格，128×128
    dim3 grid((nx)/block.x, (ny)/block.y);

    quantitize_cuda_int8<<<grid,block>>>(matrix_in,matrix_out,nx,ny,lambda);
    cudaDeviceSynchronize();
}


void dequantitize_int8(int8_t * matrix_in,float * matrix_out,int nx,int ny,float lambda){
    dim3 block(32, 32);
    //二维线程网格，128×128
    dim3 grid((nx)/block.x, (ny)/block.y);

    dequantitize_cuda_int8<<<grid,block>>>(matrix_in,matrix_out,nx,ny,lambda);
    cudaDeviceSynchronize();
}

void dequantitize_int32(int32_t * matrix_in,float * matrix_out,int nx,int ny,float lambda){
    dim3 block(32, 32);
    //二维线程网格，128×128
    dim3 grid((nx)/block.x, (ny)/block.y);

    dequantitize_cuda_int32<<<grid,block>>>(matrix_in,matrix_out,nx,ny,lambda);
    cudaDeviceSynchronize();
}


// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded 
// to avoid shared memory bank conflicts.
__global__ void strans_cuda(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void Itrans_cuda(int32_t *odata, const int32_t *idata)
{
  __shared__ int32_t tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void I8trans_cuda(int8_t *odata, const int8_t *idata)
{
  __shared__ int8_t tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}




// CUDA kernel to compute the max of elements in an array
__global__ void max_abs_in_array(float *g_idata, float *g_odata, int n) {
    // define an array in shared memory
    // the size of the array is determined by the number of threads
    extern __shared__ float sdata[];

    // get thread id
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (i < n) {
        sdata[tid] = g_idata[i]; // copy data from global mem to shared mem
    }
    __syncthreads(); // make sure all data is loaded into shared mem

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (i + s) < n) {
            sdata[tid] = max(std::abs(sdata[tid]), std::abs(sdata[tid + s]));
        }
        __syncthreads(); // make sure all adds at one stage are done!
    }

    // only thread 0 writes result back to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}


// CUDA kernel to compute the max of elements in an array
__global__ void min_abs_in_array(float *g_idata, float *g_odata, int n) {
    // define an array in shared memory
    // the size of the array is determined by the number of threads
    extern __shared__ float sdata[];

    // get thread id
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (i < n) {
        sdata[tid] = g_idata[i]; // copy data from global mem to shared mem
    }
    __syncthreads(); // make sure all data is loaded into shared mem

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (i + s) < n) {
            sdata[tid] = min(std::abs(sdata[tid]), std::abs(sdata[tid + s]));
        }
        __syncthreads(); // make sure all adds at one stage are done!
    }

    // only thread 0 writes result back to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// CUDA kernel to compute the abs sum of elements in an array
__global__ void sum_abs_in_array(float *g_idata, float *g_odata, int n) {
    // define an array in shared memory
    // the size of the array is determined by the number of threads
    extern __shared__ float sdata[];

    // get thread id
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (i < n) {
        sdata[tid] = g_idata[i]; // copy data from global mem to shared mem
    }
    __syncthreads(); // make sure all data is loaded into shared mem

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (i + s) < n) {
            sdata[tid] = std::abs(sdata[tid])+std::abs(sdata[tid + s]);
        }
        __syncthreads(); // make sure all adds at one stage are done!
    }

    // only thread 0 writes result back to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}


__global__ void quantitize_cuda_int8(float * matrix_in,int8_t * matrix_out,int nx,int ny,float lambda)
{
    int ix = threadIdx.x+blockDim.x*blockIdx.x;
    int iy = threadIdx.y+blockDim.y*blockIdx.y;
    int idx = ix+iy*ny;

    
    matrix_out[idx] = __float2int_rd(matrix_in[idx]*lambda);
}
__global__ void dequantitize_cuda_int8(int8_t * matrix_in,float * matrix_out,int nx,int ny,float lambda)
{
    int ix = threadIdx.x+blockDim.x*blockIdx.x;
    int iy = threadIdx.y+blockDim.y*blockIdx.y;
    int idx = ix+iy*ny;


    matrix_out[idx] = ((float)matrix_in[idx]/lambda);
}
__global__ void dequantitize_cuda_int32(int32_t * matrix_in,float * matrix_out,int nx,int ny,float lambda)
{
    int ix = threadIdx.x+blockDim.x*blockIdx.x;
    int iy = threadIdx.y+blockDim.y*blockIdx.y;
    int idx = ix+iy*ny;


    matrix_out[idx] = ((float)matrix_in[idx]/lambda);
}

__global__ void rowMax(float *matrix, float *row_max, int rows, int cols) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < rows) {
        float max_val = matrix[tid * cols];
        for (int i = 1; i < cols; ++i) {
            float val = matrix[tid * cols + i];
            if (val > max_val) {
                max_val = val;
            }
        }
        row_max[tid] = max_val;
    }
}

__global__ void _diag_matmul(float* A, float* x, int row, int col){

    int size = row*col;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<size){
        int j = i/col;
        A[i] = A[i] *x[j];
    }
}


void Itrans(int32_t *odata, int32_t *idata,int rows,int cols){
    dim3 dimGrid(rows/TILE_DIM, cols/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);    

    Itrans_cuda<<<dimGrid,dimBlock>>>(odata,idata);
    cudaDeviceSynchronize();
}

void I8trans(int8_t *odata, int8_t *idata,int rows,int cols){
    dim3 dimGrid(rows/TILE_DIM, cols/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);    

    I8trans_cuda<<<dimGrid,dimBlock>>>(odata,idata);
    cudaDeviceSynchronize();
}


void diag_matmul(float* A, float* x, int row, int col){
    int size = row*col;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    _diag_matmul<<<blocksPerGrid, threadsPerBlock>>>(A, x, row,col);
     
    
}


