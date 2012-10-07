package preprocessor;
use util;

########################################################################
# kernel computation
########################################################################

sub _evaluate (indata):
{
    my  ($indata) = @_;
    my $prefix='kernel_';
    my $feats=&util::get_features($indata, $prefix);
    my $fun=*{$indata->{$prefix.'name'}.'Kernel'};
    my $kargs=&util::get_args($indata, $prefix);

    $prefix='preprocessor_';
    my $pargs=&util::get_args($indata, $prefix);
    $feats=&util::add_preprocessor($indata->{$prefix.'name'}, $feats, $pargs);

    $prefix='kernel_';
    $kernel=$kfun->($feats->{'train'}, $feats->{'train'}, $kargs);
    $km_train=max(abs(
		      $indata->{$prefix.'matrix_train'}-$kernel->get_kernel_matrix())->flat);
    $kernel->init($feats->{'train'}, $feats->{'test'});
    $km_test=max(abs(
		     $indata->{$prefix.'matrix_test'}-$kernel->get_kernel_matrix())->flat);
    
    return &util::check_accuracy(
	$indata->{$prefix.'accuracy'}, {km_train=>$km_train, km_test=>$km_test});
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
Test Preprocessor
"""

from shogun.Kernel import *

import util


########################################################################
# kernel computation
########################################################################

def _evaluate (indata):
	prefix='kernel_'
	feats=util.get_features(indata, prefix)
	kfun=eval(indata[prefix+'name']+'Kernel')
	kargs=util.get_args(indata, prefix)

	prefix='preprocessor_'
	pargs=util.get_args(indata, prefix)
	feats=util.add_preprocessor(indata[prefix+'name'], feats, *pargs)

	prefix='kernel_'
	kernel=kfun(feats['train'], feats['train'], *kargs)
	km_train=max(abs(
		indata[prefix+'matrix_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	km_test=max(abs(
		indata[prefix+'matrix_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy(
		indata[prefix+'accuracy'], km_train=km_train, km_test=km_test)


########################################################################
# public
########################################################################

def test (indata):
	return _evaluate(indata)

