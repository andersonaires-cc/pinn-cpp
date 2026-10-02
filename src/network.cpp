#include "network.hpp"


PINNImpl::PINNImpl
(
    int64_t entrada_dim,
    int64_t escondida_dim,
    int64_t saida_dim
)
{
    fc1 = register_module
        (
            "fc1",
            torch::nn::Linear(
                entrada_dim,
                escondida_dim
            )
        );
    
    fc2 = register_module(
        "fc2",
        torch::nn::Linear(
            escondida_dim,
            escondida_dim
        )
    );

    fc3 = register_module(
        "fc3",
        torch::nn::Linear(
            escondida_dim,
            escondida_dim
        )
    );

    saida =
    register_module(
        "saida",
        torch::nn::Linear(
            escondida_dim,
            saida_dim
        )
    );

}

torch::Tensor PINNImpl::forward(
    torch::Tensor t
){
    auto x = torch::tanh(fc1->forward(t));

    x =
        torch::tanh(
            fc2->forward(x)
        );


    x =
        torch::tanh(
            fc3->forward(x)
        );

    return saida->forward(x);

    
}