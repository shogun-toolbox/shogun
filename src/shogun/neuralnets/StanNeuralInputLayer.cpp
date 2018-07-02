

#include <shogun/neuralnets/StanNeuralInputLayer.h>

using namespace shogun;

StanNeuralInputLayer::StanNeuralInputLayer() : StanNeuralLayer()
{
    init();
}

StanNeuralInputLayer::StanNeuralInputLayer(int32_t num_neurons, int32_t start_index):
        StanNeuralLayer(num_neurons)
{
    init();
    m_start_index = start_index;
}

StanNeuralInputLayer::StanNeuralInputLayer(int32_t width, int32_t height,
                                     int32_t num_channels, int32_t start_index): StanNeuralLayer(width*height*num_channels)
{
    init();
    m_width = width;
    m_height = height;
    m_start_index = start_index;
}

void StanNeuralInputLayer::compute_activations(StanMatrix& inputs, StanVector& parameters)
{
    auto biases = parameters.block(0,0,m_num_neurons, 1);
    StanMatrix& A = m_stan_activations;
    A.resize(m_num_neurons, m_batch_size);
    A.colwise() = biases;

    int32_t weights_index_offset = m_num_neurons;

    auto W = parameters.block(weights_index_offset, 0, m_num_neurons * inputs.rows() , 1);
    W.resize(m_num_neurons, inputs.rows());

    A += W*inputs;
}

void StanNeuralInputLayer::init()
{
    m_start_index = 0;
    gaussian_noise = 0;
    SG_ADD(&m_start_index, "start_index",
           "Start Index", MS_NOT_AVAILABLE);
    SG_ADD(&gaussian_noise, "gaussian_noise",
           "Gaussian Noise Standard Deviation", MS_NOT_AVAILABLE);
}
