#ifndef MULTICLASSSTRATEGY_H__
#define MULTICLASSSTRATEGY_H__

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
enum EMulticlassStrategy
{
	ONE_VS_REST_STRATEGY,
	ONE_VS_ONE_STRATEGY,
};
#endif

// TODO: serialization support for this, inheritance from SGObject?
class CMulticlassStrategy: public CSGObject
{
public:
	/** get number of machines used in this strategy.
	 *
	 * @param num_classes number of classes in this problem.
	 */
	virtual int32_t get_num_machines(int32_t num_classes)=0;

	/** get strategy type */
	virtual EMulticlassStrategy get_strategy_type()=0;
};

class CMulticlassOneVsRestStrategy: public CMulticlassStrategy
{
public:
	/** get number of machines used in this strategy.
	 * one-vs-rest strategy use one machine for each of the classes.
	 *
	 * @param num_classes number of classes in this problem.
	 */
	virtual int32_t get_num_machines(int32_t num_classes)
	{
		return num_classes;
	}

	/** get strategy type */
	virtual EMulticlassStrategy get_strategy_type()
	{
		return ONE_VS_REST_STRATEGY;
	}

	/** get name */
	virtual const char* get_name() const
	{
		return "MulticlassOneVsRestStrategy";
	};
};

class CMulticlassOneVsOneStrategy: public CMulticlassStrategy
{
public:
	/** get number of machines used in this strategy.
	 * one-vs-one strategy use one machine for each pair of classes.
	 *
	 * @param num_classes number of classes in this problem.
	 */
	virtual int32_t get_num_machines(int32_t num_classes)
	{
		return num_classes*(num_classes-1)/2;	
	}

	/** get strategy type */
	virtual EMulticlassStrategy get_strategy_type()
	{
		return ONE_VS_ONE_STRATEGY;
	}

	/** get name */
	virtual const char* get_name() const
	{
		return "MulticlassOneVsOneStrategy";
	};
};

} // namespace shogun

#endif /* end of include guard: MULTICLASSSTRATEGY_H__ */

