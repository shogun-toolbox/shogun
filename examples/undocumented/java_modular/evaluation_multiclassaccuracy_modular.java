import org.shogun.*;
import org.jblas.*;
import static org.jblas.DoubleMatrix.randn;

public class evaluation_multiclassaccuracy_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();

		double mul = 2.0;
		DoubleMatrix ground_truth = Load.load_labels("../data/label_train_multiclass.dat");
		DoubleMatrix predicted = Load.load_labels("../data/label_train_multiclass.dat").mmul(mul);

		MulticlassLabels ground_truth_labels = new MulticlassLabels(ground_truth);
		MulticlassLabels predicted_labels = new MulticlassLabels(predicted);

		MulticlassAccuracy evaluator = new MulticlassAccuracy();
		double accuracy = evaluator.evaluate(predicted_labels, ground_truth_labels);

		System.out.println(accuracy);

		modshogun.exit_shogun();
	}
}
