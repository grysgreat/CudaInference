#include <stdexcept>
#include "ConvLayer.hpp"
#include "Tensor.hpp"


#include <cublas_v2.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#include "helper_functions.cuh"
#include "compute_util.cuh"
#include "npy.hpp"


__global__ void make_wcol(float* w_ptr, float* res_ptr, int N, int C, int H, int W)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int filt_stride = H*W;
    int mat_stride = C*H*W;

    if (i > N){
        return;
    }
    if (j > mat_stride){
        return;
    }

    int Ni = (i);
    int Ci = (j) / (H*W);
    int Hi = (j - Ci*H*W) / W ;
    int Wi = (j - Ci*H*W - Hi*W);

    res_ptr[i*mat_stride + j] = w_ptr[Ni*mat_stride + Ci*filt_stride + Hi*W + Wi];
}

__global__ void make_imcol(float* im_ptr, float* res_ptr, int Nf, int Cf, int Hf, int Wf, int Ho, int Wo, int Hi, int Wi, int batch_size, int stride, int pad, float pad_val=0)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int i_mat_stride = Cf*Hi*Wi;
    int o_mat_stride = Cf*Hf*Wf;
    int total_scals = Ho*Wo*batch_size;

    if (i >= total_scals){
        return;
    }
    if (j >= Hf*Wf*Cf){
        return;
    }

    int Ri = ((i % (Ho*Wo)) / Wo);
    int Rj = ((i % (Ho*Wo)) % Wo);
    int ci = j / (Hf*Wf);
    int K_ind_i = (j - ci*Hf*Wf) / Wf;
    int K_ind_j = (j - ci*Hf*Wf) % Wf;

    int hi = stride*Ri + K_ind_i;
    int wi = stride*Rj + K_ind_j;
    int ni = i / (Ho*Wo); // batch

    bool is_pad = (hi < pad) || (wi < pad) || (hi >= Hi + pad) || (wi >= Wi + pad);

    if (!is_pad){
        hi -= pad;
        wi -= pad;
        res_ptr[i*o_mat_stride + j] = im_ptr[ni*i_mat_stride + ci*Hi*Wi + hi*Wi + wi];
    } else {
        res_ptr[i*o_mat_stride + j] = pad_val;
    }
}


ConvLayer::ConvLayer(cublasHandle_t& cublas_handle, const std::string& w_path, int pad, int stride, bool bias):
    cublas_handle(cublas_handle),
    _pad(pad),
    _stride(stride),
    _bias(bias),
    input_set(false)
{

    std::vector<unsigned long> shape;
    std::vector<float> data;
    bool is_f;

    npy::LoadArrayFromNumpy(w_path + ".weight.npy", shape, is_f, data);

    N = shape[0];
    C = shape[1];
    H = shape[2];
    W = shape[3];

    _w = std::shared_ptr<Tensor<float>>(new Tensor<float>({N, C, H, W}));
    _w->from_cpu(data.data());


    if (_bias){
        std::vector<unsigned long> shape_b;
        npy::LoadArrayFromNumpy(w_path + ".bias.npy", shape_b, is_f, data_b);
    }


    _wcol = std::shared_ptr<Tensor<float>>(new Tensor<float>({N, C*H*W}));

    bool weights_nchw = true;
    if (weights_nchw) {
        // weights already in im2col format
        _wcol = _w;
    } else {
        // transform weights to im2col format
        int cell_size = 32;
        dim3 block_size;
        dim3 grid_size;

        int wcol_Ho = N;
        int wcol_Wo = C*H*W;
        int num_blocks_x = wcol_Ho/cell_size + (wcol_Ho % cell_size != 0);
        int num_blocks_y = wcol_Wo/cell_size + (wcol_Wo % cell_size != 0);
        block_size = dim3(cell_size, cell_size);
        grid_size = dim3(num_blocks_x, num_blocks_y, 3);

        make_wcol<<<grid_size, block_size>>>(_w->_ptr, _wcol->_ptr, N, C, H, W);
    }
}




void ConvLayer::forward() 
{
    if (! input_set){
        throw std::runtime_error("input not set in forward");
    }

    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;
    int num_blocks_y;

    int imcol_Ho = batch_size*Ho*Wo;
    int imcol_Wo = C*H*W;
    num_blocks_x = imcol_Ho/cell_size + (imcol_Ho % cell_size != 0);
    num_blocks_y = imcol_Wo/cell_size + (imcol_Wo % cell_size != 0);
    block_size = dim3(cell_size, cell_size);
    grid_size = dim3(num_blocks_x, num_blocks_y);

    make_imcol<<<grid_size, block_size>>>(_input->_ptr, _imcol->_ptr, N, C, H, W, Ho, Wo, Hi, Wi, batch_size, _stride, _pad);
    //debug_array(_imcol->_ptr, _imcol->count());


    row_major_sgemm(cublas_handle, m, n, k, _wcol->_ptr, _imcol->_ptr, _res->_ptr, _tmp->_ptr);
    _res->reshape({N, batch_size, Ho, Wo});

    Tensor<float>::transpose(_res.get(), _tmp.get(), {1, 0, 2, 3});

    if (_bias) {
        *_tmp += *_bcol;
    }

    _res->reshape({N, batch_size*Ho*Wo});


    //debug_array(_tmp->_ptr, _tmp->count());
}


void ConvLayer::set_input(std::shared_ptr<Tensor<float>> input)
{
    if (input->size().size() != 4) {
        throw std::runtime_error("not four dims in input");
    }

    Size isize = input->size();
    batch_size = isize[0];
    Hi = isize[2];
    Wi = isize[3];
    Ho = (Hi + 2*_pad - 1*(H - 1) - 1)/_stride + 1;
    Wo = (Wi + 2*_pad - 1*(W - 1) - 1)/_stride + 1;
    m = N;
    n = batch_size*Ho*Wo;
    k = C*H*W;

    _input = input;

    _imcol = std::shared_ptr<Tensor<float>>(new Tensor<float>({C*H*W, batch_size*Ho*Wo}));

    _res = std::shared_ptr<Tensor<float>>(new Tensor<float>({N, batch_size*Ho*Wo}));
    _tmp = std::shared_ptr<Tensor<float>>(new Tensor<float>({batch_size, N, Ho, Wo}));


    if (_bias) {
        // bias array to add
        _bcol = std::shared_ptr<Tensor<float>>(new Tensor<float>({batch_size, N, Ho, Wo}));
        thrust::device_ptr<float> thr_ptr = thrust::device_pointer_cast<float>(_bcol->_ptr);
        for (int i = 0; i < batch_size; ++i){
            for (int j = 0; j < N; ++j){
                thrust::fill(thr_ptr, thr_ptr + Ho*Wo, data_b[j]);
                thr_ptr += Ho*Wo;
            }
        }
    }
    input_set = true;
}

ConvLayer::~ConvLayer(){
}

std::shared_ptr<Tensor<float>> ConvLayer::get_output()
{
    return _tmp;
}

