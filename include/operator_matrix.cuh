

float max_abs(float* d_array,float* d_work,float* c_work, int size);

float min_abs(float* d_array,float* d_work,float* c_work, int size);

float avg_abs(float* d_array,float* d_work,float* c_work, int size);

void max_abs_vec(float* d_array,float* d_work,float* c_work,float* output, int rows, int cols,char type);

void min_abs_vec(float* d_array,float* d_work,float* c_work,float* output, int rows, int cols,char type);

void avg_abs_vec(float* d_array,float* d_work,float* c_work,float* output, int rows, int cols,char type);


void strans(float *odata, float *idata,int rows,int cols);

void quantitize_int8(float * matrix_in,int8_t * matrix_out,int nx,int ny,float lambda);


void dequantitize_int8(int8_t * matrix_in,float * matrix_out,int nx,int ny,float lambda);

void dequantitize_int32(int32_t * matrix_in,float * matrix_out,int nx,int ny,float lambda);


void diag_matmul(float* A, float* x, int row, int col);

__global__ void scopy(float * matrix_in,float * matrix_out,int rows,int cols);

// __global__ void strans(float * matrix,float * result , int rows, int cols);

__global__ void strans_cuda(float *odata, const float *idata);

__global__ void max_abs_in_array(float *g_idata, float *g_odata, int n);

__global__ void min_abs_in_array(float *g_idata, float *g_odata, int n);

/**
* @brief get the sum of array
* @return g_odata: the sum of each tiny block.(add them and get the avg)
*/
__global__ void sum_abs_in_array(float *g_idata, float *g_odata, int n);


__global__ void get_min_vec(float* matrix,float* vec, int rows,int cols,char type);



__global__ void quantitize_cuda_int8(float * matrix_in,int8_t * matrix_out,int nx,int ny,float lambda);

__global__ void dequantitize_cuda_int8(int8_t * matrix_in,float * matrix_out,int nx,int ny,float lambda);

__global__ void dequantitize_cuda_int32(int32_t * matrix_in,float * matrix_out,int nx,int ny,float lambda);

__global__ void rowMax(float *matrix, float *row_max, int rows, int cols);



void Itrans(int32_t *odata, int32_t *idata,int rows,int cols);


void I8trans(int8_t *odata, int8_t *idata,int rows,int cols);