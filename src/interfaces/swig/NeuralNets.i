%shared_ptr(shogun::NeuralNetwork)
%shared_ptr(shogun::ConvolutionalFeatureMap)
%shared_ptr(shogun::RBM)
%shared_ptr(shogun::DeepBeliefNetwork)
%shared_ptr(shogun::NeuralLayers)

/* Include Class Headers to make them visible from within the target language */
RANDOM_INTERFACE(Machine)
%include <shogun/neuralnets/NeuralNetwork.h>
%include <shogun/neuralnets/NeuralLayer.h>
%include <shogun/neuralnets/ConvolutionalFeatureMap.h>
%include <shogun/neuralnets/RBM.h>
%include <shogun/neuralnets/DeepBeliefNetwork.h>
%include <shogun/neuralnets/Helpers.h>
