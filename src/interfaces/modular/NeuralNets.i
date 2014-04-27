%newobject apply(CFeatures* data);
%newobject apply_multiclass(CFeatures* data);
 
/* Remove C Prefix */
%rename(NeuralNetwork) CNeuralNetwork;
%rename(NeuralLayer) CNeuralLayer;
%rename(NeuralLinearLayer) CNeuralLinearLayer;
%rename(NeuralLogisticLayer) CNeuralLogisticLayer;
%rename(NeuralSoftmaxLayer) CNeuralSoftmaxLayer;
%rename(NeuralRectifiedLinearLayer) CNeuralRectifiedLinearLayer;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/neuralnets/NeuralNetwork.h>
%include <shogun/neuralnets/NeuralLayer.h>
%include <shogun/neuralnets/NeuralLinearLayer.h>
%include <shogun/neuralnets/NeuralLogisticLayer.h>
%include <shogun/neuralnets/NeuralSoftmaxLayer.h>
%include <shogun/neuralnets/NeuralRectifiedLinearLayer.h>
