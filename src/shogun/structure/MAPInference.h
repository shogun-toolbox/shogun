/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shell Hu, Jiaolong Xu, Bjoern Esser, Yuyu Zhang
 */

#ifndef __MAP_INFERENCE_H__
#define __MAP_INFERENCE_H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>
#include <shogun/structure/FactorGraph.h>
#include <shogun/labels/FactorGraphLabels.h>

namespace shogun
{

/** the following inference methods are acceptable:
 * Tree Max Product, Loopy Max Product, LP Relaxation,
 * Sequential Tree Reweighted Max Product (TRW-S),
 * Graph cuts
 */
enum EMAPInferType
{
	TREE_MAX_PROD = 0,
	LOOPY_MAX_PROD = 1,
	LP_RELAXATION = 2,
	TRWS_MAX_PROD = 3,
	GRAPH_CUT = 4,
	GEMP_LP = 5
};

class MAPInferImpl;

/** @brief Class MAPInference performs MAP inference on a factor graph.
 * Briefly, given a factor graph model, with features \f$\bold{x}\f$,
 * the prediction is obtained by \f$ {\arg\max} _{\bold{y}} P(\bold{Y}
 * = \bold{y} | \bold{x}; \bold{w}) \f$.
 */
class MAPInference : public SGObject
{
public:
	/** default constructor */
	MAPInference();

	/** constructor
	 *
	 * @param fg pointer of factor graph, i.e. structured inputs
	 * @param inference_method name of MAP inference method
	 */
	MAPInference(std::shared_ptr<FactorGraph> fg, EMAPInferType inference_method);

	/** destructor */
	virtual ~MAPInference();

	/** @return class name */
	virtual const char* get_name() const { return "MAPInference"; }

	/** perform inference */
	virtual void inference();

	/** get structured outputs
	 *
	 * @return FactorGraphObservation pointer
	 */
	std::shared_ptr<FactorGraphObservation> get_structured_outputs() const;

	/** @return minimized energy */
	float64_t get_energy() const;

private:
	/** register parameters and initialize members */
	void init();

protected:
	/** pointer of factor graph */
	std::shared_ptr<FactorGraph> m_fg;

	/** structured outputs */
	std::shared_ptr<FactorGraphObservation> m_outputs;

	/** minimized energy */
	float64_t m_energy;

	/** opaque pointer to hide implementation */
	std::shared_ptr<MAPInferImpl> m_infer_impl;
};

/** @brief Class CMAPInferImpl abstract class
 * of MAP inference implementation
 */
class MAPInferImpl : public SGObject
{
public:
	/** default constructor */
	MAPInferImpl();

	/** constructor
	 *
	 * @param fg pointer of factor graph, i.e. structured inputs
	 */
	MAPInferImpl(std::shared_ptr<FactorGraph> fg);

	/** destructor */
	virtual ~MAPInferImpl();

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
	std::shared_ptr<FactorGraph> m_fg;
};

}

#endif
