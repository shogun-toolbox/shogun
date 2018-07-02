

#ifndef SHOGUN_STANNEURALINPUTLAYER_H
#define SHOGUN_STANNEURALINPUTLAYER_H

#include <shogun/neuralnets/StanNeuralLayer.h>
#include <shogun/lib/common.h>

namespace shogun
{
/** @brief Represents an input layer. The layer can be either
 * connected to all the input features that a network receives (default) or
 * connected to just a small part of those features
 */
    class StanNeuralInputLayer : public StanNeuralLayer
    {
    public:
        /** default constructor */
        StanNeuralInputLayer();

        /** Constuctor
         *
         * @param num_neurons Number of neurons in this layer
         *
         * @param start_index Index of the first feature that the layer connects to,
         * i.e the activations of the layer are copied from
         * input_features[start_index:start_index+num_neurons]
         */
        StanNeuralInputLayer(int32_t num_neurons, int32_t start_index = 0);

        /** Constructs an input layer that deals with images (for convolutional nets).
         * Sets the number of neurons to width*height*num_channels
         *
         * @param width Width of the image
         *
         * @param height Width of the image
         *
         * @param num_channels Number of channels
         *
         * @param start_index Index of the first feature that the layer connects to,
         * i.e the activations of the layer are copied from
         * input_features[start_index:start_index+num_neurons]
         */
        StanNeuralInputLayer(int32_t width, int32_t height, int32_t num_channels,
                          int32_t start_index = 0);

        virtual ~StanNeuralInputLayer() {}

        /** Returns true */
        virtual bool is_input() { return true; }

        /** Copies inputs[start_index:start_index+num_neurons, :] into the
         * layer's activations
         *
         * @param inputs Input features matrix, size num_features*num_cases
         * @param parameters are the parameters of the neural network
         */
        virtual void compute_activations(StanMatrix& inputs, StanVector& parameters);

        /** Gets the index of the first feature that the layer connects to,
         * i.e the activations of the layer are copied from
         * input_features[start_index:start_index+num_neurons]
         */
        virtual int32_t get_start_index() { return m_start_index; }

        /** Sets the index of the first feature that the layer connects to,
         * i.e the activations of the layer are copied from
         * input_features[start_index:start_index+num_neurons]
         */
        virtual void set_start_index(int32_t i) { m_start_index = i; }

        virtual const char* get_name() const { return "StanNeuralInputLayer"; }

    private:
        void init();

    public:
        /** Standard deviation of the gaussian noise added to the activations of
         * the layer. Useful for denoising autoencoders. Default value is 0.0.
         */
        float64_t gaussian_noise;

    protected:
        /** Index of the first feature that the layer connects to,
         * i.e the activations of the layer are copied from
         * input_features[start_index:start_index+num_neurons]
         */
        int32_t m_start_index;
    };
}


#endif //SHOGUN_STANNEURALINPUTLAYER_H
