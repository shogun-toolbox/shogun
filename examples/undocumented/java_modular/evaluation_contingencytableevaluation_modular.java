import org.shogun.*;
import org.jblas.*;
import static org.jblas.DoubleMatrix.randn;

public class evaluation_contingencytableevaluation_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();

		DoubleMatrix ground_truth = Load.load_labels("../data/label_train_twoclass.dat");
		DoubleMatrix predicted = randn(1, ground_truth.getLength());

		BinaryLabels ground_truth_labels = new BinaryLabels(ground_truth);
		BinaryLabels predicted_labels = new BinaryLabels(predicted);

		ContingencyTableEvaluation base_evaluator = new ContingencyTableEvaluation();
		base_evaluator.evaluate(predicted_labels,ground_truth_labels);

		AccuracyMeasure evaluator1 = new AccuracyMeasure();
		double accuracy = evaluator1.evaluate(predicted_labels,ground_truth_labels);

		ErrorRateMeasure evaluator2 = new ErrorRateMeasure();
		double errorrate = evaluator2.evaluate(predicted_labels,ground_truth_labels);

		BALMeasure evaluator3 = new BALMeasure();
		double bal = evaluator3.evaluate(predicted_labels,ground_truth_labels);

		WRACCMeasure evaluator4 = new WRACCMeasure();
		double wracc = evaluator4.evaluate(predicted_labels,ground_truth_labels);

		F1Measure evaluator5 = new F1Measure();
		double f1 = evaluator5.evaluate(predicted_labels,ground_truth_labels);

		CrossCorrelationMeasure evaluator6 = new CrossCorrelationMeasure();
		double crosscorrelation = evaluator6.evaluate(predicted_labels,ground_truth_labels);

		RecallMeasure evaluator7 = new RecallMeasure();
		double recall = evaluator7.evaluate(predicted_labels,ground_truth_labels);

		PrecisionMeasure evaluator8 = new PrecisionMeasure();
		double precision = evaluator8.evaluate(predicted_labels,ground_truth_labels);

		SpecificityMeasure evaluator9 = new SpecificityMeasure();
		double specificity = evaluator9.evaluate(predicted_labels,ground_truth_labels);

		System.out.printf("%f, %f, %f, %f, %f, %f, %f, %f, %f\n", accuracy, errorrate, bal, wracc, f1, crosscorrelation, recall, precision, specificity);

		modshogun.exit_shogun();
	}
}
