%newobject apply(CFeatures* data);
%newobject apply_multiclass(CFeatures* data);
%newobject visible_state_features();
%newobject sample_group(int32_t V, int32_t num_gibbs_steps, int32_t batch_size);
%newobject sample_group_with_evidence(int32_t V, int32_t E, CDenseFeatures<float64_t>* evidence,int32_t num_gibbs_steps);
%newobject reconstruct(CDenseFeatures<float64_t>* data);
%newobject transform(CDenseFeatures<float64_t>* data);
%newobject done();

/* Remove C Prefix */
%rename(NeuralNetwork) CNeuralNetwork;
%rename(NeuralLayer) CNeuralLayer;
%rename(NeuralInputLayer) CNeuralInputLayer;
%rename(NeuralLinearLayer) CNeuralLinearLayer;
%rename(NeuralLogisticLayer) CNeuralLogisticLayer;
%rename(NeuralSoftmaxLayer) CNeuralSoftmaxLayer;
%rename(NeuralRectifiedLinearLayer) CNeuralRectifiedLinearLayer;
%rename(ConvolutionalFeatureMap) CConvolutionalFeatureMap;
%rename(NeuralConvolutionalLayer) CNeuralConvolutionalLayer;
%rename(RBM) CRBM;
%rename(DeepBeliefNetwork) CDeepBeliefNetwork;
%rename(Autoencoder) CAutoencoder;
%rename(DeepAutoencoder) CDeepAutoencoder;
%rename(NeuralLayers) CNeuralLayers;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/neuralnets/NeuralNetwork.h>
%include <shogun/neuralnets/NeuralLayer.h>
%include <shogun/neuralnets/NeuralInputLayer.h>
%include <shogun/neuralnets/NeuralLinearLayer.h>
%include <shogun/neuralnets/NeuralLogisticLayer.h>
%include <shogun/neuralnets/NeuralSoftmaxLayer.h>
%include <shogun/neuralnets/NeuralRectifiedLinearLayer.h>
%include <shogun/neuralnets/ConvolutionalFeatureMap.h>
%include <shogun/neuralnets/NeuralConvolutionalLayer.h>
%include <shogun/neuralnets/RBM.h>
%include <shogun/neuralnets/DeepBeliefNetwork.h>
%include <shogun/neuralnets/Autoencoder.h>
%include <shogun/neuralnets/DeepAutoencoder.h>
%include <shogun/neuralnets/NeuralLayers.h>
