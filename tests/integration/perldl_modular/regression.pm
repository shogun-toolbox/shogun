package regression;
use util;
use PDL;
use modshogun;

sub _evaluate (indata):
{
    my ($indata) = @_;
    my $prefix = 'kernel_';
    my $feats = &util::get_features($indata, $prefix);
    my $kargs = &util::get_args($indata, $prefix);
    my $fun = eval('modshogun::' . $indata->{$prefix.'name'}.'Kernel');
    my $kernel = $fun->new($feats->{'train'}, $feats->{'train'}, @$kargs);

    $prefix='regression_';
    $kernel->{parallel}->set_num_threads($indata->{$prefix.'num_threads'});
    my $rfun = eval('modshogun::' . $indata->{$prefix.'name'});
    if($@) {#except NameError, e:
	warn( "%s is disabled/unavailable!",$indata->{$prefix.'name'});
	return false;
    }
    my $labels = modshogun::RegressionLabels->new($indata->{$prefix.'labels'});
    if($indata->{$prefix.'type'} eq 'svm') {
	$regression = $rfun->new(
	    $indata->{$prefix.'C'}, $indata->{$prefix.'epsilon'}, $kernel, $labels);
    }elsif($indata->{$prefix.'type'} eq 'kernelmachine') {
	$regression = $rfun->new($indata->{$prefix.'tau'}, $kernel, $labels);
    }else{
	return false;
    }
    $regression->{parallel}->set_num_threads($indata->{$prefix.'num_threads'});
    if(defined($indata->{$prefix.'tube_epsilon'})) {
	$regression->set_tube_epsilon($indata->{$prefix.'tube_epsilon'});
    }
    $regression->train();
    
    my $alphas=0;
    my $bias=0;
    my $sv=0;
    if(defined($indata->{$prefix.'bias'})) {
	$bias = abs($regression->get_bias()-$indata->{$prefix.'bias'});
    }
    if(defined($indata->{$prefix.'alphas'})) {
	foreach my $item (@{ $regression->get_alphas()->tolist()}) {
	    $alphas+=$item;
	}
	$alphas=abs($alphas-$indata->{$prefix.'alphas'});
    }
    if(defined($indata->{$prefix.'support_vectors'})){
	foreach my $item (@{$inregression->get_support_vectors()->tolist()}) {
	    $sv+=$item;
	}
	$sv=abs($sv - $indata->{$prefix.'support_vectors'});
    }
    $kernel->init($feats->{'train'}, $feats->{'test'});
    my $classified=max(abs(
			   $regression->apply()->get_labels()-$indata->{$prefix.'classified'}));
    
    return &util::check_accuracy($indata->{$prefix.'accuracy'}
				 , {alphas=>$alphas,
				    bias=>$bias, support_vectors=>$sv, classified=>$classified});
}
########################################################################
# public
########################################################################
sub test
{
    my ($indata) = @_;
    return &_evaluate($indata);
}
true;
__END__
"""
Test Regression
"""

from numpy import double
from shogun.Features import RegressionLabels
from shogun.Kernel import *
from shogun.Regression import *

import util

def _evaluate (indata):
	prefix='kernel_'
	feats=util.get_features(indata, prefix)
	kargs=util.get_args(indata, prefix)
	fun=eval(indata[prefix+'name']+'Kernel')
	kernel=fun(feats['train'], feats['train'], *kargs)

	prefix='regression_'
	kernel.parallel.set_num_threads(indata[prefix+'num_threads'])

	try:
		rfun=eval(indata[prefix+'name'])
	except NameError, e:
		print "%s is disabled/unavailable!"%indata[prefix+'name']
		return False

	labels=RegressionLabels(double(indata[prefix+'labels']))
	if indata[prefix+'type']=='svm':
		regression=rfun(
			indata[prefix+'C'], indata[prefix+'epsilon'], kernel, labels)
	elif indata[prefix+'type']=='kernelmachine':
		regression=rfun(indata[prefix+'tau'], kernel, labels)
	else:
		return False

	regression.parallel.set_num_threads(indata[prefix+'num_threads'])
	if indata.has_key(prefix+'tube_epsilon'):
		regression.set_tube_epsilon(indata[prefix+'tube_epsilon'])

	regression.train()

	alphas=0
	bias=0
	sv=0
	if indata.has_key(prefix+'bias'):
		bias=abs(regression.get_bias()-indata[prefix+'bias'])
	if indata.has_key(prefix+'alphas'):
		for item in regression.get_alphas().tolist():
			alphas+=item
		alphas=abs(alphas-indata[prefix+'alphas'])
	if indata.has_key(prefix+'support_vectors'):
		for item in inregression.get_support_vectors().tolist():
			sv+=item
		sv=abs(sv-indata[prefix+'support_vectors'])

	kernel.init(feats['train'], feats['test'])
	classified=max(abs(
		regression.apply().get_labels()-indata[prefix+'classified']))

	return util.check_accuracy(indata[prefix+'accuracy'], alphas=alphas,
		bias=bias, support_vectors=sv, classified=classified)

########################################################################
# public
########################################################################

def test (indata):
	return _evaluate(indata)

