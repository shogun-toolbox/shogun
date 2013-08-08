/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu 
 * Copyright (C) 2013 Shell Hu 
 */

#ifndef __MAP_INFERENCE_H__
#define __MAP_INFERENCE_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>
#include <shogun/structure/FactorGraph.h>
#include <shogun/structure/FactorGraphLabels.h>

namespace shogun
{

/** the following inference methods are acceptable: 
 * Tree Max Product, Loopy Max Product, LP Relaxation, 
 * Sequential Tree Reweighted Max Product (TRW-S),
 * Iterated Conditional Mode (ICM), Naive Mean Field,
 * Structured Mean Field.
 */
enum EMAPInferType
{
	TREE_MAX_PROD = 0,
	LOOPY_MAX_PROD = 1,
	LP_RELAXATION = 2,
	TRWS_MAX_PROD = 3,
	ITER_COND_MODE = 4,
	NAIVE_MEAN_FIELD = 5,
	STRUCT_MEAN_FIELD = 6,
};

class CMAPInferImpl;

/** @brief Class CMAPInference performs MAP inference on a factor graph.
 * Briefly, given a factor graph model, with features \f$\bold{x}\f$,
 * the prediction is obtained by \f$ {\arg\max} _{\bold{y}} P(\bold{Y} 
 * = \bold{y} | \bold{x}; \bold{w}) \f$.
 */
class CMAPInference : public CSGObject
{
public:
	/** default constructor */
	CMAPInference();

	/** constructor
	 *
	 * @param fg pointer of factor graph, i.e. structured inputs
	 * @param inference_method name of MAP inference method
	 */
	CMAPInference(CFactorGraph* fg, EMAPInferType inference_method);

	/** destructor */
	virtual ~CMAPInference();

	/** @return class name */
	virtual const char* get_name() const { return "MAPInference"; }

	/** perform inference */
	virtual void inference();

	/** get structured outputs 
	 *
	 * @return CFactorGraphObservation pointer 
	 */
	virtual CFactorGraphObservation* get_structured_outputs() const;

	/** @return minimized energy */
	float64_t get_energy() const;

private:
	/** register parameters and initialize members */
	void init();

protected:
	/** pointer of factor graph */
	CFactorGraph* m_fg; 

	/** structured outputs */
	CFactorGraphObservation* m_outputs;

	/** minimized energy */
	float64_t m_energy;

	/** opaque pointer to hide implementation */
	CMAPInferImpl* m_infer_impl; 
};

/** @brief Class CMAPInferImpl abstract class 
 * of MAP inference implementation 
 */
class CMAPInferImpl : public CSGObject
{
public:
	/** default constructor */
	CMAPInferImpl();

	/** constructor
	 *
	 * @param fg pointer of factor graph, i.e. structured inputs
	 */
	CMAPInferImpl(CFactorGraph* fg);

	/** destructor */
	virtual ~CMAPInferImpl();

	/** @return class name */
	virtual const char* get_name() const { return "MAPInferImpl"; }

	/** perform inference (need to be implemented)
	 *
	 * @param assignment outputs of inference results
	 */
	virtual float64_t inference(SGVector<int32_t> assignment) = 0;

private:
	/** register parameters */
	void register_parameters();

protected:
	/** pointer of factor graph */
	CFactorGraph* m_fg;
};

}

#endif
