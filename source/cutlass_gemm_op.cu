#include "cutlass_gemm_op.cuh"


void cut_gemm(input_t *A, input_t* B, output_t* C,int rowA,int colA, int rowB,int colB){

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(rowA, rowB, colA);


  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);  


  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     {A,colA},  // <- reference to matrix A on device
                                     {B,colA},  // <- reference to matrix B on device
                                     {C,colB},  // <- reference to matrix C on device
                                     {C,colB},  // <- reference to matrix D on device
                                     {alpha, beta}       // <- tuple of alpha and beta
                                     };        // <- k-dimension split factor


  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<input_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  gemm_op();

  cudaDeviceSynchronize();


    // for (int i = 0; i < 10; ++i) {
    //     for (int j = 0; j < 10; ++j) {
    //         // 访问并打印元素
    //         std::cout << static_cast<int>(tensor_c.at({i, j})) << " ";
    //     }
    //     std::cout << std::endl; // 每行后换行
    // }

}

