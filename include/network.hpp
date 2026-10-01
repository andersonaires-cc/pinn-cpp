#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <torch/torch.h>

class PINNImpl : public torch::nn::Module{

    public:
        PINNImpl(
            int64_t entrada_dim1 = 1,
            int64_t escondida_dim = 32,
            int64_t saida_dim = 1

        );

         torch::Tensor forward(
            torch::Tensor t
        );

    private:

        torch::nn::Linear fc1{nullptr};

        torch::nn::Linear fc2{nullptr};

        torch::nn::Linear fc3{nullptr};

        torch::nn::Linear saida{nullptr};
    
};

TORCH_MODULE(PINN);

#endif