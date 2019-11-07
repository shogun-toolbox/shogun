/* Remove C Prefix */
%shared_ptr(shogun::NeuralNetwork)
%shared_ptr(shogun::NeuralLayer)
%shared_ptr(shogun::NeuralInputLayer)
%shared_ptr(shogun::NeuralLinearLayer)
%shared_ptr(shogun::NeuralLogisticLayer)
%shared_ptr(shogun::NeuralSoftmaxLayer)
%shared_ptr(shogun::NeuralRectifiedLinearLayer)
%shared_ptr(shogun::ConvolutionalFeatureMap)
%shared_ptr(shogun::NeuralConvolutionalLayer)
%shared_ptr(shogun::RBM)
%shared_ptr(shogun::DeepBeliefNetwork)
%shared_ptr(shogun::Autoencoder)
%shared_ptr(shogun::DeepAutoencoder)
%shared_ptr(shogun::NeuralLayers)

/* Include Class Headers to make them visible from within the target language */
%include <shogun/neuralnets/NeuralLayer.h>
%include <shogun/neuralnets/ConvolutionalFeatureMap.h>
%include <shogun/neuralnets/RBM.h>
%include <shogun/neuralnets/DeepBeliefNetwork.h>
%include <shogun/neuralnets/Autoencoder.h>
%include <shogun/neuralnets/DeepAutoencoder.h>
