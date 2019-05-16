%newobject apply(CFeatures* data);
%newobject apply_multiclass(CFeatures* data);
%newobject visible_state_features();
%newobject sample_group(int32_t V, int32_t num_gibbs_steps, int32_t batch_size);
%newobject sample_group_with_evidence(int32_t V, int32_t E, CDenseFeatures<float64_t>* evidence,int32_t num_gibbs_steps);
%newobject reconstruct(CDenseFeatures<float64_t>* data);
%newobject transform(CDenseFeatures<float64_t>* data);
%newobject done();
%newobject input(int32_t size);
%newobject logistic(int32_t size);
%newobject linear(int32_t size);
%newobject rectified_linear(int32_t size);
%newobject leaky_rectified_linear(int32_t size);
%newobject softmax(int32_t size);

/* Remove C Prefix */
%rename(ConvolutionalFeatureMap) CConvolutionalFeatureMap;
%rename(NeuralLayer) CNeuralLayer;
%rename(RBM) CRBM;
%rename(DeepBeliefNetwork) CDeepBeliefNetwork;
%rename(Autoencoder) CAutoencoder;
%rename(DeepAutoencoder) CDeepAutoencoder;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/neuralnets/NeuralLayer.h>
%include <shogun/neuralnets/ConvolutionalFeatureMap.h>
%include <shogun/neuralnets/RBM.h>
%include <shogun/neuralnets/DeepBeliefNetwork.h>
%include <shogun/neuralnets/Autoencoder.h>
%include <shogun/neuralnets/DeepAutoencoder.h>
