import org.shogun.*;
import org.jblas.*;
import static org.jblas.DoubleMatrix.randn;

public class evaluation_prcevaluation_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();

		DoubleMatrix ground_truth = Load.load_labels("../data/label_train_twoclass.dat");
		DoubleMatrix predicted = randn(1, ground_truth.getLength());

		BinaryLabels ground_truth_labels = new BinaryLabels(ground_truth);
		BinaryLabels predicted_labels = new BinaryLabels(predicted);

		PRCEvaluation evaluator = new PRCEvaluation();
		evaluator.evaluate(predicted_labels, ground_truth_labels);

		System.out.println(evaluator.get_PRC());
		System.out.println(evaluator.get_auPRC());

	}
}
