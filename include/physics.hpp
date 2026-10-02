#ifndef PHYSICS_HPP
#define PHYSICS_HPP

#include <torch/torch.h>


class Physics {

public:

    static torch::Tensor residual(
        torch::Tensor u,
        torch::Tensor d2u_dt2
    );

};


#endif