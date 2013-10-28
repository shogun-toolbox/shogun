###########################################################################
# Mean prediction from Gaussian Processes based on
# classifier_libsvm_minimal_modular.py
# plotting functions have been adapted from the pyGP library
# https://github.com/jameshensman/pyGP
###########################################################################
from numpy import *
from numpy.random import randn
from modshogun import *
from modshogun import *
from modshogun import *
from modshogun import *
import pylab as PL
import matplotlib
import logging as LG
import scipy as SP
from modshogun import GradientModelSelection
from modshogun import ModelSelectionParameters, R_EXP, R_LINEAR
from modshogun import ParameterCombination



def plot_training_data(x, y,
                       shift=None,
                       replicate_indices=None,
                       format_data={'alpha':.5,
                                    'marker':'.',
                                    'linestyle':'--',
                                    'lw':1,
                                    'markersize':9},
                       draw_arrows=0,
                       plot_old=False):
    """
    Plot training data input x and output y into the
    active figure (See http://matplotlib.sourceforge.net/ for details of figure).

    Instance plot without replicate groups:

    .. image:: ../images/plotTraining.png
       :height: 8cm

    Instance plot with two replicate groups and a shift in x-koords:

    .. image:: ../images/plotTrainingShiftX.png
       :height: 8cm

    **Parameters:**

    x : [double]
        Input x (e.g. time).

    y : [double]
        Output y (e.g. expression).

    shift : [double]
        The shift of each replicate group.

    replicate_indices : [int]
        Indices of replicates for each x, rexpectively

    format_data : {format}
        Format of the data points. See http://matplotlib.sourceforge.net/ for details.

    draw_arrows : int
        draw given number of arrows (if greator than len(replicate) draw all arrows.
        Arrows will show the time shift for time points, respectively.

    """
    x_shift = SP.array(x.copy())

    if shift is not None and replicate_indices is not None:
        assert len(shift) == len(SP.unique(replicate_indices)), 'Need one shift per replicate to plot properly'

        _format_data = format_data.copy()
        if(format_data.has_key('alpha')):
            _format_data['alpha'] = .2*format_data['alpha']
        else:
            _format_data['alpha'] = .2

        number_of_groups = len(SP.unique(replicate_indices))

        for i in SP.unique(replicate_indices):
            x_shift[replicate_indices == i] -= shift[i]

        for i in SP.unique(replicate_indices):
            col = matplotlib.cm.jet(i / (2. * number_of_groups))
            _format_data['color'] = col
            if(plot_old):
                PL.plot(x[replicate_indices == i], y[replicate_indices == i], **_format_data)
            if(draw_arrows):
                range = SP.where(replicate_indices == i)[0]
                for n in SP.arange(range[0], range[-1], max(1, round(len(range) / draw_arrows))):
                    offset = round((len(range)-1) / draw_arrows)
                    n += max(int((i+1)*offset/number_of_groups),1)
                    PL.text((x_shift[n]+x[n])/2., y[n],
                            "%.2f"%(-shift[i]),
                            ha='center',va='center',
                            fontsize=10)
                    PL.annotate('', xy=(x_shift[n], y[n]),
                                xytext=(x[n], y[n]),va='center',
                                arrowprops=dict(facecolor=col,
                                                alpha=.2,
                                                shrink=.01,
                                                frac=.2,
                                                headwidth=11,
                                                width=11))
            #PL.plot(x,y,**_format_data)



    if(replicate_indices is not None):
        number_of_groups = len(SP.unique(replicate_indices))
        #format_data['markersize'] = 13
        #format_data['alpha'] = .5
        for i in SP.unique(replicate_indices):
            col = matplotlib.cm.jet(i / (2. * number_of_groups))
            format_data['color'] = col
            PL.plot(x_shift[replicate_indices == i], y[replicate_indices == i], **format_data)
    else:
        print(x_shift.shape)
	number_of_groups = x_shift.shape[0]
	for i in xrange(number_of_groups):
	    col = matplotlib.cm.jet(i / (2. * number_of_groups))
            format_data['color'] = col
	    PL.plot(x[i], y[i], **format_data)

#    return PL.plot(x_shift,y,**format_data)

def plot_sausage(X, mean, std, alpha=None, format_fill={'alpha':0.3, 'facecolor':'k'}, format_line=dict(alpha=1, color='g', lw=3, ls='dashed')):
    """
plot saussage plot of GP. I.e:

.. image:: ../images/sausage.png
:height: 8cm

**returns:** : [fill_plot, line_plot]
The fill and the line of the sausage plot. (i.e. green line and gray fill of the example above)
**Parameters:**

X : [double]
Interval X for which the saussage shall be plottet.
mean : [double]
The mean of to be plottet.
std : [double]
Pointwise standard deviation.

format_fill : {format}
The format of the fill. See http://matplotlib.sourceforge.net/ for details.

format_line : {format}
The format of the mean line. See http://matplotlib.sourceforge.net/ for details.
"""
    X = X.squeeze()
    Y1 = (mean + 2 * std)
    Y2 = (mean - 2 * std)

    if(alpha is not None):
        old_alpha_fill = min(1, format_fill['alpha'] * 2)
        for i, a in enumerate(alpha[:-2]):
            format_fill['alpha'] = a * old_alpha_fill
            hf = PL.fill_between(X[i:i + 2], Y1[i:i + 2], Y2[i:i + 2], lw=0, **format_fill)
        i += 1
        hf = PL.fill_between(X[i:], Y1[i:], Y2[i:], lw=0, **format_fill)
    else:
        hf = PL.fill_between(X, Y1, Y2, **format_fill)
    hp = PL.plot(X, mean, **format_line)
    return [hf, hp]


class CrossRect(matplotlib.patches.Rectangle):
    def __init__(self, *args, **kwargs):
        matplotlib.patches.Rectangle.__init__(self, *args, **kwargs)

        #self.ax = ax

    # def get_verts(self):
    #     rectverts = matplotlib.patches.Rectangle.get_verts(self)

    #     return verts

    def get_path(self, *args, **kwargs):
        old_path = matplotlib.patches.Rectangle.get_path(self)
        verts = []
        codes = []
        for vert, code in old_path.iter_segments():
            verts.append(vert)
            codes.append(code)
        verts.append([1, 1])
        codes.append(old_path.LINETO)
        new_path = matplotlib.artist.Path(verts, codes)
        return new_path

def create_toy_data():
    #0. generate Toy-Data; just samples from a superposition of a sin + linear trend
    xmin = 1
    xmax = 2.5*SP.pi
    x = SP.arange(xmin,xmax,(xmax-xmin)/100.0)

    C = 2       #offset
    sigma = 0.5

    b = 0

    y  = b*x + C + 1*SP.sin(x)
#    dy = b   +     1*SP.cos(x)
    y += sigma*random.randn(y.shape[0])

    y-= y.mean()

    x = x[:,SP.newaxis]
    return [x,y]


def run_demo():
    LG.basicConfig(level=LG.INFO)
    random.seed(572)

    #1. create toy data
    [x,y] = create_toy_data()

    feat_train = RealFeatures(transpose(x));
    labels = RegressionLabels(y);

    n_dimensions = 1

    #2. location of unispaced predictions
    X = SP.linspace(0,10,10)[:,SP.newaxis]

    #new interface with likelihood parametres being decoupled from the covaraince function
    likelihood = GaussianLikelihood()
    covar_parms = SP.log([2])
    hyperparams = {'covar':covar_parms,'lik':SP.log([1])}

    #construct covariance function
    SECF = GaussianKernel(feat_train, feat_train,2)
    covar = SECF
    zmean = ZeroMean();
    inf = ExactInferenceMethod(SECF, feat_train, zmean, labels, likelihood);

    gp = GaussianProcessRegression(inf, feat_train, labels);

    root=ModelSelectionParameters();
    c1=ModelSelectionParameters("inference_method", inf);
    root.append_child(c1);

    c2 = ModelSelectionParameters("scale");
    c1.append_child(c2);
    c2.build_values(0.01, 4.0, R_LINEAR);
    c3 = ModelSelectionParameters("likelihood_model", likelihood);
    c1.append_child(c3);

    c4=ModelSelectionParameters("sigma");
    c3.append_child(c4);
    c4.build_values(0.001, 4.0, R_LINEAR);
    c5 =ModelSelectionParameters("kernel", SECF);
    c1.append_child(c5);

    c6 =ModelSelectionParameters("width");
    c5.append_child(c6);
    c6.build_values(0.001, 4.0, R_LINEAR);

    crit = GradientCriterion();

    grad=GradientEvaluation(gp, feat_train, labels,
			crit);

    grad.set_function(inf);

    gp.print_modsel_params();

    root.print_tree();

    grad_search=GradientModelSelection(
			root, grad);

    grad.set_autolock(0);

    best_combination=grad_search.select_model(1);

    gp.set_return_type(GaussianProcessRegression.GP_RETURN_COV);

    St = gp.apply_regression(feat_train);

    St = St.get_labels();

    gp.set_return_type(GaussianProcessRegression.GP_RETURN_MEANS);

    M = gp.apply_regression();

    M = M.get_labels();

    #create plots
    plot_sausage(transpose(x),transpose(M),transpose(SP.sqrt(St)));
    plot_training_data(x,y);
    PL.show();

if __name__ == '__main__':
    run_demo()
