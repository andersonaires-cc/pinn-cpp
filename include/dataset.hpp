#ifndef DATASET_HPP
#define DATASET_HPP

#include <torch/torch.h>


struct PINNData {
    torch::Tensor t_fisica;
    torch::Tensor t_inicial;

};

class Dataset {

public:

    static PINNData gerar(
        int num_pontos
    );

};



#endif