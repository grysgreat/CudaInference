
#include <cublas_v2.h>
#include "helper_functions.cuh"

#include "cutlass_gemm_op.cuh"
#include "operator_matrix.cuh"
#include <omp.h>

#include <rsvd/Prelude.hpp>

#include <Eigen/Dense>
#include <array>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <unordered_map>

#include <iostream> 
#include <algorithm> 
const int max_omp_thread = omp_get_max_threads();

template <typename T>
void get_R(T matrix_in[],T matrix_cmp[],T matrixR[],int rows,int cols){
    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            matrixR[i*cols+j] = (matrix_in[i*cols+j] - matrix_cmp[i*cols+j]);
        }
    }
}
template <typename T>
void xcopy(T matrix1[],T matrix2[], int rows, int cols ) {
    #pragma omp parallel for num_threads(max_omp_thread)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
			matrix2[i*cols+j] = matrix1[i*cols+j];
        }
    }
}
template <typename T>
void xmadd(T* matrixA,T* matrixB,T* matrixC,int rows,int cols){
    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            matrixC[i*cols+j] = matrixA[i*cols+j] + matrixB[i*cols+j];
        }
    }
}
void eigen_copy(int *A ,Eigen::MatrixXi &matrix_A, int rowA,int colA){

    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0;i<rowA;i++){
        for(int j=0;j<colA;j++){
            matrix_A(i,j) = A[i*colA+j];
            //printf("%d\t", matrix_A(i,j));
        }
    }
}
void eigen_copy(Eigen::MatrixXi matrix_A, int *A ,int rowA,int colA){

    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0;i<rowA;i++){
        for(int j=0;j<colA;j++){
            A[i*colA+j]=matrix_A(i,j);
        }
    }
}
void eigen_copy(float *A ,Eigen::MatrixXf &matrix_A, int rowA,int colA){

    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0;i<rowA;i++){
        for(int j=0;j<colA;j++){
            matrix_A(i,j) = A[i*colA+j];
        }
    }
}
void eigen_copy(Eigen::MatrixXf matrix_A, float *A ,int rowA,int colA){

    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0;i<rowA;i++){
        for(int j=0;j<colA;j++){
            A[i*colA+j]=matrix_A(i,j);
        }
    }
}
void eigen_SGEMM(float *A ,float *B ,float *C, int rowA,int colA, int rowB,int colB){
    Eigen::setNbThreads(1);

    Eigen::MatrixXf matrix_A{rowA, colA};
    Eigen::MatrixXf matrix_B{rowB, colB};

    eigen_copy(A , matrix_A, rowA,colA);
    eigen_copy(B , matrix_B, rowB,colB);
    Eigen::MatrixXf matrix_C = matrix_A*matrix_B;

    // printMatrix_eigen(matrix_A, rowA,colA);
    // printMatrix_eigen( matrix_B, rowB,colB);

    eigen_copy(matrix_C , C, rowA,colB);


    return;
}
void eigen_IGEMM(int *A ,int *B ,int *C, int rowA,int colA, int rowB,int colB){
    Eigen::setNbThreads(1);

    Eigen::MatrixXi matrix_A{rowA, colA};
    Eigen::MatrixXi matrix_B{rowB, colB};

    eigen_copy(A , matrix_A, rowA,colA);
    eigen_copy(B , matrix_B, rowB,colB);
    Eigen::MatrixXi matrix_C = matrix_A*matrix_B;

    // printMatrix_eigen(matrix_A, rowA,colA);
    // printMatrix_eigen( matrix_B, rowB,colB);

    eigen_copy(matrix_C , C, rowA,colB);


    return;
}


void svd_df(float A[],  float A_L[],float A_R[], int rowsA, int colsA, int rank) {

  const int numRowsA{rowsA};
  const int numColsA{colsA};
  const int reducedRank{rank};

  Eigen::MatrixXf matrix_X{numRowsA, numColsA};
  eigen_copy(A,matrix_X,rowsA,colsA);

  // Randomized SVD
  std::mt19937_64 randomEngine{};
  randomEngine.seed(777);
  Rsvd::RandomizedSvd<Eigen::MatrixXf, std::mt19937_64, Rsvd::SubspaceIterationConditioner::Lu> rsvdx(
      randomEngine);
  rsvdx.compute(matrix_X, reducedRank,1,1);


  const Eigen::MatrixXf rsvdApprox_A_UE{rsvdx.matrixU()*rsvdx.singularValues().asDiagonal() };
  const Eigen::MatrixXf rsvdApprox_A_V{rsvdx.matrixV().adjoint()};
  eigen_copy(rsvdApprox_A_UE,A_L,numRowsA,reducedRank);   
  eigen_copy(rsvdApprox_A_V,A_R,reducedRank,numColsA);   

  const Eigen::MatrixXf rsvdApprox_A_U{rsvdx.matrixU()};
  const Eigen::MatrixXf rsvdApprox_A_Diagonal{rsvdx.singularValues().asDiagonal()};



}


template <typename T>
void xtrans(T matrix[],T result[] , int rows, int cols) {


    T* tmp =(T *)malloc(sizeof(T) * rows*cols);
    #pragma omp parallel for num_threads(max_omp_thread)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            int originalIndex = i * cols + j;

            int transposedIndex = j * rows + i;
            tmp[transposedIndex] = matrix[originalIndex];
        }
    }
    xcopy<T>(tmp,result,cols,rows);
}

template <typename T>
T get_max(T* matrix,int rows,int cols){
    T maxM=0;

    #pragma omp parallel for reduction(max:maxM)
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){

            maxM = std::max(maxM,std::abs(matrix[i*cols+j]));
        }
    }
    return maxM;
}

template <typename T>
T get_max_s(T* matrix,int rows,int cols){
    T maxM=0;

    #pragma omp parallel for reduction(max:maxM)
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){

            maxM = std::max(maxM,(matrix[i*cols+j]));
        }
    }
    return maxM;
}


template <typename T>
T get_min_s(T* matrix,int rows,int cols){
    T maxM=100000;

    #pragma omp parallel for reduction(max:maxM)
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){

            maxM = std::min(maxM,(matrix[i*cols+j]));
        }
    }
    return maxM;
}


template <typename T>
void get_max_vec(T* matrix,T* vec, int rows,int cols,char type){
    
    if(type == 'r'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int i=0;i<rows;i++){
            T maxM=0;
            for(int j=0;j<cols;j++){
                maxM = std::max(std::abs(maxM),std::abs(matrix[i*cols+j]));
            }
            vec[i] = maxM;
        }
    }
    if(type == 'c'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int j=0;j<cols;j++){
            T maxM=0;
            for(int i=0;i<rows;i++){
                maxM = std::max(std::abs(maxM),std::abs(matrix[i*cols+j]));
            }
            vec[j] = maxM;
        }
    }    
}

template <typename T>
void get_lambda_vec(T* lambda_vec,T* max_vec,int digit, int len){


    int max_int = (1<<(digit-1)) - 1;
    // #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0;i<len;i++){
        lambda_vec[i]=(T)max_int/max_vec[i];
    }
}

template <typename Ti,typename To>
void quantitize_vec(Ti* matrix_in,To* matrix_out, float* lambda_vec,int rows,int cols,char type,char rc){

    if(type == 'q'){
        if(rc == 'r'){
            #pragma omp parallel for num_threads(max_omp_thread)
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    matrix_out[i*cols+j] = (int)(std::floor(matrix_in[i*cols+j]*lambda_vec[i]));
                }
            }
        }
        else if(rc == 'c'){
            #pragma omp parallel for num_threads(max_omp_thread)
            for(int i=0;i<cols;i++){
                for(int j=0;j<rows;j++){
                    matrix_out[i+j*cols] = (int)(std::floor(matrix_in[i+j*cols]*lambda_vec[i]));
                }
            }        
        }
    }
    else if(type == 'd'){
        if(rc == 'r'){
            #pragma omp parallel for num_threads(max_omp_thread)
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    matrix_out[i*cols+j] = (To)matrix_in[i*cols+j]/lambda_vec[i];
                }
            }
        }
        else if(rc == 'c'){
            #pragma omp parallel for num_threads(max_omp_thread)
            for(int i=0;i<cols;i++){
                for(int j=0;j<rows;j++){
                    matrix_out[i+j*cols] = (To)(matrix_in[i+j*cols]/lambda_vec[i]);
                }
            }        
        }    
    }
    return;
}

template <typename Ti,typename To>
void quantitize_vec0(Ti* matrix_in,To* matrix_out, float* lambda_vec,int rows,int cols,char type,char rc){

    if(type == 'q'){
        if(rc == 'r'){
            #pragma omp parallel for num_threads(max_omp_thread)
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    matrix_out[i*cols+j] = (int)((matrix_in[i*cols+j]*lambda_vec[i]));
                }
            }
        }
        else if(rc == 'c'){
            #pragma omp parallel for num_threads(max_omp_thread)
            for(int i=0;i<cols;i++){
                for(int j=0;j<rows;j++){
                    matrix_out[i+j*cols] = (int)((matrix_in[i+j*cols]*lambda_vec[i]));
                }
            }        
        }
    }
    else if(type == 'd'){
        if(rc == 'r'){
            #pragma omp parallel for num_threads(max_omp_thread)
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    matrix_out[i*cols+j] = (To)matrix_in[i*cols+j]/lambda_vec[i];
                }
            }
        }
        else if(rc == 'c'){
            #pragma omp parallel for num_threads(max_omp_thread)
            for(int i=0;i<cols;i++){
                for(int j=0;j<rows;j++){
                    matrix_out[i+j*cols] = (To)(matrix_in[i+j*cols]/lambda_vec[i]);
                }
            }        
        }    
    }
    return;
}

template <typename Ti,typename To>
void dequantitize_matrix(Ti* matrix_in,To* matrix_out, float* lambda_r,  float* lambda_c,int rows,int cols){
    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            matrix_out[i*cols+j] = (To)(matrix_in[i*cols+j]/(lambda_r[i]*lambda_c[j]));
        }
    }
}

template <typename Ti,typename To>
void quantitize(Ti* matrix_in,To* matrix_out,int rows,int cols,float lambda,char type){

    if(type == 'q'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                matrix_out[i*cols+j] = (To)((int)(matrix_in[i*cols+j]*lambda));
                //printf("%d\t",(int)(matrix_in[i*cols+j]*lambda));
            }
            //printf("\n");
        }
    }
    else if(type == 'd'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                matrix_out[i*cols+j] = ((To)matrix_in[i*cols+j])/lambda;
            }
        }        
    }
}


template <typename Ti,typename To>
void quantitize_floor(Ti* matrix_in,To* matrix_out,int rows,int cols,float lambda,char type){

    if(type == 'q'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                matrix_out[i*cols+j] = (To)((int)(std::floor(matrix_in[i*cols+j]*lambda)));
                //printf("%f, %d\t",matrix_in[i*cols+j]*lambda, matrix_out[i*cols+j]);
            }
            //printf("\n");
        }
    }
    else if(type == 'd'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                matrix_out[i*cols+j] = ((To)matrix_in[i*cols+j])/lambda;
            }
        }        
    }
}




template <typename T>
void xgemm(const T A[], const T B[], T C[], int rowsA, int colsA, int rowsB, int colsB) {


    T* tmp =(T *)malloc(sizeof(T) * rowsA*colsB);
    #pragma omp parallel for num_threads(max_omp_thread)
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            tmp[i * colsB + j] = 0;
            for (int k = 0; k < colsA; ++k) {
                tmp[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
    #pragma omp parallel for num_threads(max_omp_thread)
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            C[i * colsB + j]  = tmp[i * colsB + j];
		}
	}
    
}


// 定义矩阵乘矩阵函数---- svd法
template <typename T,int digit>
void xigemm(T *A, T *B, T *C, int rowsA, int colsA, int rowsB, int colsB) {

    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    
    int *A_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *B_int = (int *)malloc(sizeof(int) * rowsB*colsB);
    int *C_int = (int *)malloc(sizeof(int) * rowsA*colsB);


    T *C_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);

    const int max_int = (1<<(digit-1)) - 1;
    T max_mA =get_max<T>(A,rowsA, colsA);
    T max_mB =get_max<T>(B,rowsB, colsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;

    
    //对A,B矩阵进行直接量化
    quantitize<T,int>(A,A_int,rowsA, colsA,lambdaA,'q');
    quantitize<T,int>(B,B_int,rowsB, colsB,lambdaB,'q');

    xgemm(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    T lambdaC = lambdaA*lambdaB;
    quantitize<int,T>(C_int,C_buffer,rowsA,colsB,lambdaC,'d');

    xcopy<T>(C_buffer,C, rowsA, colsB);
}

// 完全残差矩阵乘法
template <typename T,int digit>
void rxigemm(T *A, T *B, T *C, int rowsA, int colsA, int rowsB, int colsB) {

    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    
    int *A_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *B_int = (int *)malloc(sizeof(int) * rowsB*colsB);
    int *C_int = (int *)malloc(sizeof(int) * rowsA*colsB);
    int *AR_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *BR_int = (int *)malloc(sizeof(int) * rowsB*colsB);

    T *C_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *C_copy = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_p = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_p = (T *)malloc(sizeof(T) * rowsB * colsB);


    const int max_int = (1<<(digit-1)) - 1;
    T max_mA =get_max<T>(A,rowsA, colsA);
    T max_mB =get_max<T>(B,rowsB, colsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;

    //对A,B矩阵进行直接量化
    quantitize<T,int>(A,A_int,rowsA, colsA,lambdaA,'q');
    quantitize<T,int>(B,B_int,rowsB, colsB,lambdaB,'q');

    eigen_IGEMM(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    //对结果矩阵C_INT进行反量化得到C'
    T lambdaC = lambdaA*lambdaB;
    quantitize<int,T>(C_int,C_buffer,rowsA,colsB,lambdaC,'d');
    
    //Dequantization of the quantized int matrix gives A',B'
    quantitize<int,T>(A_int,A_p,rowsA,colsA,lambdaA,'d');
    quantitize<int,T>(B_int,B_p,rowsB,colsB,lambdaB,'d');

    //Calculate the residuals matrix for full size
    get_R<T>(A,A_p,A_p,rowsA,colsA);
    get_R<T>(B,B_p,B_p,rowsB,colsB);

    //The residual matrix is quantized
    T max_mAR =get_max<T>(A_p,rowsA,colsA);
    T max_mBR =get_max<T>(B_p,rowsB,colsB);

    T lambdaAR = (T)max_int/max_mAR;
    T lambdaBR = (T)max_int/max_mBR;
    //Direct quantization of A and B residual matrices
    quantitize<T,int>(A_p,AR_int,rowsA,colsA,lambdaAR,'q');
    quantitize<T,int>(B_p,BR_int,rowsB,colsB,lambdaBR,'q');

    //The error repair matrix float is obtained by inverse quantization of the int error repair matrix
    T lambdaCR1 = lambdaA*lambdaBR;
    T lambdaCR2 = lambdaAR*lambdaB;
    eigen_IGEMM(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
    quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
    //The error is supplemented by a repair matrix
    xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

    eigen_IGEMM(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR2,'d');
    //The error is supplemented by a repair matrix
    xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

    xcopy<T>(C_buffer,C, rowsA, colsB);

}

// 完全残差矩阵乘法
template <typename T,int digit>
void rxigemm2(T *A, T *B, T *C, int rowsA, int colsA, int rowsB, int colsB) {

    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    
    int *A_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *B_int = (int *)malloc(sizeof(int) * rowsB*colsB);
    int *C_int = (int *)malloc(sizeof(int) * rowsA*colsB);
    int *AR_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *BR_int = (int *)malloc(sizeof(int) * rowsB*colsB);

    T *C_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *C_copy = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_p = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_p = (T *)malloc(sizeof(T) * rowsB * colsB);


    const int max_intA = (1<<(4-1)) - 1;;
    const int max_intB = (1<<(5-1)) - 1;;
    T max_mA =get_max<T>(A,rowsA, colsA);
    T max_mB =get_max<T>(B,rowsB, colsB);
    T lambdaA = (T)max_intA/max_mA;
    T lambdaB = (T)max_intB/max_mB;

    //对A,B矩阵进行直接量化
    quantitize<T,int>(A,A_int,rowsA, colsA,lambdaA,'q');
    quantitize<T,int>(B,B_int,rowsB, colsB,lambdaB,'q');

    eigen_IGEMM(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    //对结果矩阵C_INT进行反量化得到C'
    T lambdaC = lambdaA*lambdaB;
    quantitize<int,T>(C_int,C_buffer,rowsA,colsB,lambdaC,'d');
    
    //Dequantization of the quantized int matrix gives A',B'
    quantitize<int,T>(A_int,A_p,rowsA,colsA,lambdaA,'d');
    quantitize<int,T>(B_int,B_p,rowsB,colsB,lambdaB,'d');

    //Calculate the residuals matrix for full size
    get_R<T>(A,A_p,A_p,rowsA,colsA);
    get_R<T>(B,B_p,B_p,rowsB,colsB);

    //The residual matrix is quantized
    T max_mAR =get_max<T>(A_p,rowsA,colsA);
    T max_mBR =get_max<T>(B_p,rowsB,colsB);

    T lambdaAR = (T)max_intA/max_mAR;
    T lambdaBR = (T)max_intB/max_mBR;
    //Direct quantization of A and B residual matrices
    quantitize<T,int>(A_p,AR_int,rowsA,colsA,lambdaAR,'q');
    quantitize<T,int>(B_p,BR_int,rowsB,colsB,lambdaBR,'q');

    //The error repair matrix float is obtained by inverse quantization of the int error repair matrix
    T lambdaCR1 = lambdaA*lambdaBR;
    T lambdaCR2 = lambdaAR*lambdaB;
    eigen_IGEMM(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
    quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
    //The error is supplemented by a repair matrix
    xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

    eigen_IGEMM(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR2,'d');
    //The error is supplemented by a repair matrix
    xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

    xcopy<T>(C_buffer,C, rowsA, colsB);

}

// 完全残差矩阵乘法
template <typename T,int digit>
void rfxigemm(T *A, T *B, T *C, int rowsA, int colsA, int rowsB, int colsB) {

    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    
    int *A_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *B_int = (int *)malloc(sizeof(int) * rowsB*colsB);
    int *C_int = (int *)malloc(sizeof(int) * rowsA*colsB);
    int *AR_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *BR_int = (int *)malloc(sizeof(int) * rowsB*colsB);

    T *C_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *C_copy = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_p = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_p = (T *)malloc(sizeof(T) * rowsB * colsB);


    const int max_int = (1<<(digit-1)) - 1;
    T max_mA =get_max<T>(A,rowsA, colsA);
    T max_mB =get_max<T>(B,rowsB, colsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;

    //对A,B矩阵进行直接量化
    quantitize<T,int>(A,A_int,rowsA, colsA,lambdaA,'q');
    quantitize<T,int>(B,B_int,rowsB, colsB,lambdaB,'q');

    eigen_IGEMM(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    //对结果矩阵C_INT进行反量化得到C'
    T lambdaC = lambdaA*lambdaB;
    quantitize<int,T>(C_int,C_buffer,rowsA,colsB,lambdaC,'d');
    
    //Dequantization of the quantized int matrix gives A',B'
    quantitize<int,T>(A_int,A_p,rowsA,colsA,lambdaA,'d');
    quantitize<int,T>(B_int,B_p,rowsB,colsB,lambdaB,'d');

    //Calculate the residuals matrix for full size
    get_R<T>(A,A_p,A_p,rowsA,colsA);
    get_R<T>(B,B_p,B_p,rowsB,colsB);

    //The residual matrix is quantized
    T max_mAR =get_max<T>(A_p,rowsA,colsA);
    T max_mBR =get_max<T>(B_p,rowsB,colsB);

    T lambdaAR = (T)max_int/max_mAR;
    T lambdaBR = (T)max_int/max_mBR;
    //Direct quantization of A and B residual matrices
    quantitize<T,int>(A_p,AR_int,rowsA,colsA,lambdaAR,'q');
    quantitize<T,int>(B_p,BR_int,rowsB,colsB,lambdaBR,'q');

    //The error repair matrix float is obtained by inverse quantization of the int error repair matrix
    T lambdaCR1 = lambdaA*lambdaBR;
    T lambdaCR2 = lambdaAR*lambdaB;
    T lambdaCR3 = lambdaAR*lambdaBR;
    eigen_IGEMM(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
    quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
    //The error is supplemented by a repair matrix
    xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

    eigen_IGEMM(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
    quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR2,'d');
    //The error is supplemented by a repair matrix
    xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

    eigen_IGEMM(AR_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
    quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR3,'d');
    xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

    xcopy<T>(C_buffer,C, rowsA, colsB);

}

template <typename T,int digit>
void xigemm_vec(T *A, T *B, T *C, int rowsA, int colsA, int rowsB, int colsB) {

    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    
    int *A_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *B_int = (int *)malloc(sizeof(int) * rowsB*colsB);
    int *C_int = (int *)malloc(sizeof(int) * rowsA*colsB);

    T *C_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *C_copy = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_p = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_p = (T *)malloc(sizeof(T) * rowsB * colsB);
    T *A_R = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_R = (T *)malloc(sizeof(T) * rowsB * colsB);

    T *tmp_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);

    const int max_int = (1<<(digit-1)) - 1;
    T* max_A_r = (T *)malloc(sizeof(T) * rowsA);
    get_max_vec<T>(A,max_A_r,rowsA, colsA,'r');
    T* max_B_c = (T *)malloc(sizeof(T) * colsB);
    get_max_vec<T>(B,max_B_c,rowsB, colsB,'c');
    T* lambdaA_vec = (T *)malloc(sizeof(T) * rowsA); 
    T* lambdaB_vec = (T *)malloc(sizeof(T) * colsB);

    get_lambda_vec(lambdaA_vec,max_A_r, digit, rowsA);
    get_lambda_vec(lambdaB_vec,max_B_c, digit, colsB);

    //Direct quantization of A and B matrices
    quantitize_vec0<T,int>(A,A_int,lambdaA_vec ,rowsA,colsA,'q','r');
    quantitize_vec0<T,int>(B,B_int,lambdaB_vec ,rowsB,colsB,'q','c');

    eigen_IGEMM(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    dequantitize_matrix<int,T>(C_int,C,lambdaA_vec,lambdaB_vec,rowsA,colsB);


}

template <typename T,int digit>
void xigemm_vec2(T *A, T *B, T *C, int rowsA, int colsA, int rowsB, int colsB) {

    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    
    int *A_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *B_int = (int *)malloc(sizeof(int) * rowsB*colsB);
    int *C_int = (int *)malloc(sizeof(int) * rowsA*colsB);

    T *C_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *C_copy = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_p = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_p = (T *)malloc(sizeof(T) * rowsB * colsB);
    T *A_R = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_R = (T *)malloc(sizeof(T) * rowsB * colsB);

    T *tmp_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);

    T* max_A_r = (T *)malloc(sizeof(T) * rowsA);
    get_max_vec<T>(A,max_A_r,rowsA, colsA,'r');
    T* max_B_c = (T *)malloc(sizeof(T) * colsB);
    get_max_vec<T>(B,max_B_c,rowsB, colsB,'c');
    T* lambdaA_vec = (T *)malloc(sizeof(T) * rowsA); 
    T* lambdaB_vec = (T *)malloc(sizeof(T) * colsB);

    get_lambda_vec(lambdaA_vec,max_A_r, 4, rowsA);
    get_lambda_vec(lambdaB_vec,max_B_c, 5, colsB);

    //Direct quantization of A and B matrices
    quantitize_vec0<T,int>(A,A_int,lambdaA_vec ,rowsA,colsA,'q','r');
    quantitize_vec0<T,int>(B,B_int,lambdaB_vec ,rowsB,colsB,'q','c');

    eigen_IGEMM(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    dequantitize_matrix<int,T>(C_int,C,lambdaA_vec,lambdaB_vec,rowsA,colsB);


}


template <typename T>
T get_Ferror(T matrix_ref[],T matrix_cmp[],int rows,int cols){

    T sumR=0,sum=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            sumR+=(matrix_ref[i*cols+j] - matrix_cmp[i*cols+j])*(matrix_ref[i*cols+j] - matrix_cmp[i*cols+j]);
            sum+=(matrix_ref[i*cols+j])*(matrix_ref[i*cols+j]);
        }
    }

    T ans = 0;
    ans = sqrt(sumR)/sqrt(sum);
    return ans;

}

template <typename T>
T get_Mean(T matrix_ref[],int rows,int cols){

    double sumR=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            sumR+= matrix_ref[i*cols+j];
        }
    }

    T ans = 0;
    ans = sumR/(float(rows*cols));
    return ans;

}

// 完全残差矩阵乘法
template <typename T,int digit>
void lrxigemm(T *A, T *B, T *C, int rowsA, int colsA, int rowsB, int colsB, int rank) {

    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    
    int *A_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *B_int = (int *)malloc(sizeof(int) * rowsB*colsB);
    int *C_int = (int *)malloc(sizeof(int) * rowsA*colsB);

    T *C_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *C_copy = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_p = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_p = (T *)malloc(sizeof(T) * rowsB * colsB);
    T *A_R = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_R = (T *)malloc(sizeof(T) * rowsB * colsB);

    T *A_RL = (T *)malloc(sizeof(T) * rowsA * rank);
    T *A_RR = (T *)malloc(sizeof(T) * rank * colsA);

    T *B_RL = (T *)malloc(sizeof(T) * rowsB * rank);
    T *B_RR = (T *)malloc(sizeof(T) * rank * colsB);

    T *tmp_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);

    const int max_int = (1<<(digit-1)) - 1;
    T max_mA =get_max<T>(A,rowsA, colsA);
    T max_mB =get_max<T>(B,rowsB, colsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;

    //对A,B矩阵进行直接量化
    quantitize_floor<T,int>(A,A_int,rowsA, colsA,lambdaA,'q');
    quantitize_floor<T,int>(B,B_int,rowsB, colsB,lambdaB,'q');

    eigen_IGEMM(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    //对结果矩阵C_INT进行反量化得到C'
    T lambdaC = lambdaA*lambdaB;
    quantitize<int,T>(C_int,C_buffer,rowsA,colsB,lambdaC,'d');
    
    //Dequantization of the quantized int matrix gives A',B'
    quantitize<int,T>(A_int,A_p,rowsA,colsA,lambdaA,'d');
    quantitize<int,T>(B_int,B_p,rowsB,colsB,lambdaB,'d');

    //Calculate the residuals matrix for full size
    get_R<T>(A,A_p,A_R,rowsA,colsA);
    get_R<T>(B,B_p,B_R,rowsB,colsB);

    svd_df(A_R,A_RL,A_RR,rowsA, colsA,rank);
    svd_df(B_R,B_RL,B_RR,rowsB, colsB,rank);


    eigen_SGEMM(A_RR,B_p,tmp_buffer,rank,colsA,rowsB,colsB);
    eigen_SGEMM(A_RL,tmp_buffer,C_copy,rowsA,rank,rank,colsB);
    xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


    eigen_SGEMM(A_p,B_RL,tmp_buffer,rowsA,colsA,rowsB,rank);
    eigen_SGEMM(tmp_buffer,B_RR,C_copy,rowsA,rank,rank,colsB);
    xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);



    eigen_SGEMM(A_RL,A_RR,A_R,rowsA,rank,rank,colsA);
    eigen_SGEMM(A_R,B_RL,tmp_buffer,rowsA,colsA,rowsB,rank);
    eigen_SGEMM(tmp_buffer,B_RR,C_copy,rowsA,rank,rank,colsB);
    xmadd<float>(C_copy,C_buffer,C,rowsA,colsB);


}




// 完全残差矩阵乘法
template <typename T,int digit>
void lrxigemm_vec(T *A, T *B, T *C, int rowsA, int colsA, int rowsB, int colsB, int rank) {

    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    
    int *A_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *B_int = (int *)malloc(sizeof(int) * rowsB*colsB);
    int *C_int = (int *)malloc(sizeof(int) * rowsA*colsB);

    T *C_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *C_copy = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_p = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_p = (T *)malloc(sizeof(T) * rowsB * colsB);
    T *A_R = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_R = (T *)malloc(sizeof(T) * rowsB * colsB);

    T *A_RL = (T *)malloc(sizeof(T) * rowsA * rank);
    T *A_RR = (T *)malloc(sizeof(T) * rank * colsA);

    T *B_RL = (T *)malloc(sizeof(T) * rowsB * rank);
    T *B_RR = (T *)malloc(sizeof(T) * rank * colsB);

    T *tmp_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);

    const int max_int = (1<<(digit-1)) - 1;
    T* max_A_r = (T *)malloc(sizeof(T) * rowsA);
    get_max_vec<T>(A,max_A_r,rowsA, colsA,'r');
    T* max_B_c = (T *)malloc(sizeof(T) * colsB);
    get_max_vec<T>(B,max_B_c,rowsB, colsB,'c');
    T* lambdaA_vec = (T *)malloc(sizeof(T) * rowsA); 
    T* lambdaB_vec = (T *)malloc(sizeof(T) * colsB);

    get_lambda_vec(lambdaA_vec,max_A_r, digit, rowsA);
    get_lambda_vec(lambdaB_vec,max_B_c, digit, colsB);

    //Direct quantization of A and B matrices
    quantitize_vec<T,int>(A,A_int,lambdaA_vec ,rowsA,colsA,'q','r');
    quantitize_vec<T,int>(B,B_int,lambdaB_vec ,rowsB,colsB,'q','c');

    eigen_IGEMM(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    dequantitize_matrix<int,T>(C_int,C_buffer,lambdaA_vec,lambdaB_vec,rowsA,colsB);



    
    //Dequantization of the quantized int matrix gives A',B'
    quantitize_vec<int,T>(A_int,A_p,lambdaA_vec ,rowsA,colsA,'d','r');
    quantitize_vec<int,T>(B_int,B_p,lambdaB_vec ,rowsB,colsB,'d','c');

    //Calculate the residuals matrix for full size
    get_R<T>(A,A_p,A_R,rowsA,colsA);
    get_R<T>(B,B_p,B_R,rowsB,colsB);

    svd_df(A_R,A_RL,A_RR,rowsA, colsA,rank);
    svd_df(B_R,B_RL,B_RR,rowsB, colsB,rank);


    eigen_SGEMM(A_RR,B_p,tmp_buffer,rank,colsA,rowsB,colsB);
    eigen_SGEMM(A_RL,tmp_buffer,C_copy,rowsA,rank,rank,colsB);
    xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


    eigen_SGEMM(A_p,B_RL,tmp_buffer,rowsA,colsA,rowsB,rank);
    eigen_SGEMM(tmp_buffer,B_RR,C_copy,rowsA,rank,rank,colsB);
    xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);



    //eigen_SGEMM(A_RL,A_RR,A_R,rowsA,rank,rank,colsA);
    eigen_SGEMM(A_R,B_RL,tmp_buffer,rowsA,colsA,rowsB,rank);
    eigen_SGEMM(tmp_buffer,B_RR,C_copy,rowsA,rank,rank,colsB);
    xmadd<float>(C_copy,C_buffer,C,rowsA,colsB);



    //printf("\nR1-R3 = %f,%f,%f\n",R1,R2,R3);
    // xcopy<T>(C_buffer,C, rowsA, colsB);

}



// 完全残差矩阵乘法
template <typename T,int digit>
void lrxigemm_vec2(T *A, T *B, T *C, int rowsA, int colsA, int rowsB, int colsB, int rank) {

    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    
    int *A_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *B_int = (int *)malloc(sizeof(int) * rowsB*colsB);
    int *C_int = (int *)malloc(sizeof(int) * rowsA*colsB);

    T *C_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *C_copy = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_p = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_p = (T *)malloc(sizeof(T) * rowsB * colsB);
    T *A_R = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_R = (T *)malloc(sizeof(T) * rowsB * colsB);

    T *A_RL = (T *)malloc(sizeof(T) * rowsA * rank);
    T *A_RR = (T *)malloc(sizeof(T) * rank * colsA);

    T *B_RL = (T *)malloc(sizeof(T) * rowsB * rank);
    T *B_RR = (T *)malloc(sizeof(T) * rank * colsB);

    T *tmp_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);

    T* max_A_r = (T *)malloc(sizeof(T) * rowsA);
    get_max_vec<T>(A,max_A_r,rowsA, colsA,'r');
    T* max_B_c = (T *)malloc(sizeof(T) * colsB);
    get_max_vec<T>(B,max_B_c,rowsB, colsB,'c');
    T* lambdaA_vec = (T *)malloc(sizeof(T) * rowsA); 
    T* lambdaB_vec = (T *)malloc(sizeof(T) * colsB);

    get_lambda_vec(lambdaA_vec,max_A_r, 4, rowsA);
    get_lambda_vec(lambdaB_vec,max_B_c, 5, colsB);

    //Direct quantization of A and B matrices
    quantitize_vec<T,int>(A,A_int,lambdaA_vec ,rowsA,colsA,'q','r');
    quantitize_vec<T,int>(B,B_int,lambdaB_vec ,rowsB,colsB,'q','c');

    eigen_IGEMM(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    dequantitize_matrix<int,T>(C_int,C_buffer,lambdaA_vec,lambdaB_vec,rowsA,colsB);



    
    //Dequantization of the quantized int matrix gives A',B'
    quantitize_vec<int,T>(A_int,A_p,lambdaA_vec ,rowsA,colsA,'d','r');
    quantitize_vec<int,T>(B_int,B_p,lambdaB_vec ,rowsB,colsB,'d','c');

    //Calculate the residuals matrix for full size
    get_R<T>(A,A_p,A_R,rowsA,colsA);
    get_R<T>(B,B_p,B_R,rowsB,colsB);

    svd_df(A_R,A_RL,A_RR,rowsA, colsA,rank);
    svd_df(B_R,B_RL,B_RR,rowsB, colsB,rank);


    eigen_SGEMM(A_RR,B_p,tmp_buffer,rank,colsA,rowsB,colsB);
    eigen_SGEMM(A_RL,tmp_buffer,C_copy,rowsA,rank,rank,colsB);
    xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


    eigen_SGEMM(A_p,B_RL,tmp_buffer,rowsA,colsA,rowsB,rank);
    eigen_SGEMM(tmp_buffer,B_RR,C_copy,rowsA,rank,rank,colsB);
    xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);



    eigen_SGEMM(A_RL,A_RR,A_R,rowsA,rank,rank,colsA);
    eigen_SGEMM(A_R,B_RL,tmp_buffer,rowsA,colsA,rowsB,rank);
    eigen_SGEMM(tmp_buffer,B_RR,C_copy,rowsA,rank,rank,colsB);
    xmadd<float>(C_copy,C_buffer,C,rowsA,colsB);



    //printf("\nR1-R3 = %f,%f,%f\n",R1,R2,R3);
    // xcopy<T>(C_buffer,C, rowsA, colsB);

}

__global__ void debug_ker(float* ptr, int addr){
    //int i = blockIdx.x*blockDim.x + threadIdx.x;
    printf("%d %f\n", addr, ptr[addr]);
}

void debug_array(float* arr, int N){
    for (int i = 0; i < N; ++i){
        debug_ker<<<1,1>>>(arr, i);
    }
    cudaDeviceSynchronize();
}

// ./Release/cuda_proj --input /root/CV/dataset/val/n01440764/ILSVRC2012_val_00009111.ppm --weights_dir ../python/weights/ --batch_size 1 --iters 1
void row_major_sgemm(cublasHandle_t& cublas_handle, int m, int n, int k, float* A, float* B, float* C, float* tmp){
    float alpha = 1.0;
    float beta = 0.0;

    float *A_H = (float *)malloc(sizeof(float) * m*k);
    float *B_H = (float *)malloc(sizeof(float) * k*n);
    float *C_H = (float *)malloc(sizeof(float) * m*n);

    float *D_H = (float *)malloc(sizeof(float) * m*n);



    float* tmp2;
    cudaMalloc((void**)&tmp2, sizeof(float) * n*k);
    cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, k, &alpha, B, k, &beta, tmp2, n, tmp2, n);
    cudaDeviceSynchronize();

    cudaMemcpy( A_H,A, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
    cudaMemcpy( B_H,tmp2, sizeof(float) * k * n, cudaMemcpyDeviceToHost);
    float meanA = get_Mean(A_H,m,k);
    float meanB = get_Mean(B_H,k,n);
    float max_mA =get_max<float>(A_H,m,k);
    float max_mB =get_max<float>(B_H,k,n);

    float max_mAs =get_max_s<float>(A_H,m,k);
    float max_mBs =get_max_s<float>(B_H,k,n);

    float min_mAs =get_min_s<float>(A_H,m,k);
    float min_mBs =get_min_s<float>(B_H,k,n);

    int min_mkn = std::min(m,n);
    min_mkn = std::min(min_mkn,k);



    //xigemm<float,8>(A_H,B_H,C_H,m,k,k,n);
    //xigemm_vec2<float,5>(A_H,B_H,C_H,m,k,k,n);
    rfxigemm<float,4>(A_H,B_H,C_H,m,k,k,n);
    //lrxigemm_vec<float,8>(A_H,B_H,C_H,m,k,k,n,min_mkn/10);
    //<float,8>(A_H,B_H,C_H,m,k,k,n,20);

    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, k, B, k, &beta, tmp, m));
    checkCublasErrors(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, tmp, m, &beta, C, n, C, n));
    cudaDeviceSynchronize();
    
    cudaMemcpy( D_H,C, sizeof(float) * n * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, C_H, sizeof(float) * m*n, cudaMemcpyHostToDevice);



    float R3 = get_Ferror<float>(D_H,C_H,m,n); 
    printf("%.4f\n",R3);

    // printf("ERR=%.4f,mnk= %d,%d,%d, mean = %f,%f,max =%f,%f, max-s =%f,%f , min-s =%f,%f\n",
    //     R3,m,n,k,meanA,meanB,max_mA,max_mB, max_mAs,max_mBs, min_mAs,min_mBs);

}

void row_major_sgemm_add(cublasHandle_t& cublas_handle, int m, int n, int k, float* A, float* B, float*D, float* C, float* tmp){
    float alpha = 1.0;
    float beta = 0.0;

    float *A_H = (float *)malloc(sizeof(float) * m*k);
    float *B_H = (float *)malloc(sizeof(float) * k*n);
    float *C_H = (float *)malloc(sizeof(float) * m*n);

    float *D_H = (float *)malloc(sizeof(float) * m*n);



    // float* tmp2;
    // cudaMalloc((void**)&tmp2, sizeof(float) * n*k);
    // cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, k, &alpha, B, k, &beta, tmp2, n, tmp2, n);
    // cudaDeviceSynchronize();

    // cudaMemcpy( A_H,A, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
    // cudaMemcpy( B_H,tmp2, sizeof(float) * k * n, cudaMemcpyDeviceToHost);
    // float meanA = get_Mean(A_H,m,k);
    // float meanB = get_Mean(B_H,k,n);
    // float max_mA =get_max<float>(A_H,m,k);
    // float max_mB =get_max<float>(B_H,k,n);

    // printf("ADD- mnk= %d,%d,%d, mean = %f,%f,max =%f,%f\n",m,n,k,meanA,meanB,max_mA,max_mB);


    
    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, k, B, k, &beta, tmp, m));
    beta = 1.0;
    checkCublasErrors(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, tmp, m, &beta, D, n, C, n));
}


template<typename T>
__global__ void add_ker(T* src1, T* src2, T* dst, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N){
        return;
    }
    dst[i] = src1[i] + src2[i];
}

template<typename T>
void cuda_add(T* src1, T* src2, T* res, int N){
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    num_blocks_x = N/cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);
    add_ker<T><<<grid_size, block_size>>>(src1, src2, res, N);
}

template<typename T>
__global__ void sub_ker(T* src1, T* src2, T* dst, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N){
        return;
    }
    dst[i] = src1[i] - src2[i];
}

template<typename T>
void cuda_sub(T* src1, T* src2, T* res, int N){
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    num_blocks_x = N/cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);
    sub_ker<T><<<grid_size, block_size>>>(src1, src2, res, N);
}


template<typename T>
__global__ void mul_ker(T* src1, T* src2, T* dst, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N){
        return;
    }
    dst[i] = src1[i] * src2[i];
}

template<typename T>
void cuda_mul(T* src1, T* src2, T* res, int N){
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    num_blocks_x = N/cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);
    mul_ker<T><<<grid_size, block_size>>>(src1, src2, res, N);
}

template<typename T>
__global__ void div_ker(T* src1, T* src2, T* dst, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N){
        return;
    }
    dst[i] = src1[i] / src2[i];
}

template<typename T>
void cuda_div(T* src1, T* src2, T* res, int N){
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    num_blocks_x = N/cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);
    div_ker<T><<<grid_size, block_size>>>(src1, src2, res, N);
}


template<typename T>
__global__ void transpose_ker(T* src_ptr, T* dst_ptr, int* src_dims, int* strides, int* reorder, int* new_strides, int Ndims, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    //if (i == 0){
    //    printf("%d \n", total);
    //}
    if (i >= N){
        return;
    }

    int new_idx[10];
    int acc = 0;
    for (int k = 0; k < Ndims; ++k) {
        int cur_i = (i - acc) / strides[k];
        acc += cur_i*strides[k];

        new_idx[reorder[k]] = cur_i;
    }

    int new_i = 0;
    for (int k = 0; k < Ndims; ++k) {
        new_i += new_strides[k]*new_idx[k];
    }

    dst_ptr[new_i] = src_ptr[i];
}

template<typename T>
void cuda_transpose(T* src_ptr, T* dst_ptr, int* src_dims, int* strides, int* reorder, int* new_strides, int Ndims, int N){
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;
    num_blocks_x = (N)/cell_size + ((N) % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);

    transpose_ker<<<grid_size, block_size>>>(src_ptr, dst_ptr, src_dims, strides, reorder, new_strides, Ndims, N);
}


template void cuda_add<float>(float*, float*, float*, int);
template void cuda_sub<float>(float*, float*, float*, int);
template void cuda_mul<float>(float*, float*, float*, int);
template void cuda_div<float>(float*, float*, float*, int);

template void cuda_add<int>(int*, int*, int*, int);
template void cuda_sub<int>(int*, int*, int*, int);
template void cuda_mul<int>(int*, int*, int*, int);
template void cuda_div<int>(int*, int*, int*, int);

template void cuda_transpose<float>(float*, float*, int*, int*, int*, int*, int, int);
template void cuda_transpose<int>(int*, int*, int*, int*, int*, int*, int, int);
