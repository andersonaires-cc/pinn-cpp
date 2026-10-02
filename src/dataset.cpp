#include "dataset.hpp"


PINNData Dataset::gerar(
    int num_pontos
){
    PINNData dados;

    dados.t_fisica =
        torch::linspace(
            0.0f,
            2.0f * 3.14159265f,
            num_pontos
        )
        .view({-1, 1});

    dados.t_inicial =
        torch::tensor(
            {{0.0f}}
        );


    return dados;
}