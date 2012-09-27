/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __STREAMINGDATAGENERATOR_H_
#define __DATAGENERATOR_H_
#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/streaming/StreamingFeatures.h>

namespace shogun
{

enum EDataGenerator
{
	DG_NONE, DG_DENSE_MEAN, DG_DENSE_ROT_SYM_GAUSS
};

/** Abstract base class for data generation with streaming features interface.
 * Override get_streamed_data() method in order to create new generators.
 *
 * This class provides an interface for setting the model of the data generator.
 * These models may be defined in the EDataGenerator enum. Implementations
 * of this class can check for different models and then perform corresponding
 * actions. Along with the model type come model parameters. These are provided
 * in an instance of the Parameter class (use Parameter::add() methods).
 */
class CStreamingDataGenerator: public CStreamingFeatures
{
public:
	/** Constructor */
	CStreamingDataGenerator();

	/** Destructor */
	virtual ~CStreamingDataGenerator();

	/** Sets a new underlyong data generation model and its parameters
	 *
	 * @param model model to use
	 * @param model_parameters Parameter instance to use as parameters.
	 * See model descriptions for necessary parameters. Will complain if
	 * wrong type or missing parameters. Is deleted in desctructor or when
	 * being replaced so do not use elsewhere.
	 */
	void set_model(EDataGenerator model, Parameter* model_parameters);

	/** @return name of SG_SERIALIZABLE */
	inline virtual const char* get_name() const=0;

private:
	/** registers all parameters and initializes variables with defaults */
	void init();

protected:
	/** model of data to generate */
	EDataGenerator m_model;

	/** parameters of model */
	Parameter* m_model_parameters;

};

}

#endif /* __STREAMINGDATAGENERATOR_H_ */
