#!/usr/bin/env python
parameter_list = [[1000]]

def evaluation_thresholds_modular (index):
	from modshogun import BinaryLabels, ROCEvaluation
	import numpy
	numpy.random.seed(17)
	output=numpy.arange(-1,1,0.001)
	output=(0.3*output+0.7*(numpy.random.rand(len(output))-0.5))
	label=[-1.0]*(len(output)//2)
	label.extend([1.0]*(len(output)//2))
	label=numpy.array(label)

	pred=BinaryLabels(output)
	truth=BinaryLabels(label)

	evaluator=ROCEvaluation()
	evaluator.evaluate(pred, truth)

	[fp,tp]=evaluator.get_ROC()

	thresh=evaluator.get_thresholds()
	b=thresh[index]

	#print("tpr", numpy.mean(output[label>0]>b), tp[index])
	#print("fpr", numpy.mean(output[label<0]>b), fp[index])

	return tp[index],fp[index],numpy.mean(output[label>0]>b),numpy.mean(output[label<0]>b)

if __name__=='__main__':
	print('Evaluation with Thresholds')
	evaluation_thresholds_modular(*parameter_list[0])
