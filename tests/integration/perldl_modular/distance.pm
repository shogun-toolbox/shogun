package distance;
use modshogun;
#use shogun::Distance;

use util;
use PDL;

sub _evaluate {
    my ($indata) = @_;
    my $prefix ='distance_';
    my $feats = &util::get_features($indata, $prefix);
    
    my $dfun = eval('modshogun::' . $indata->{$prefix.'name'});

    my $dargs = &util::get_args($indata, $prefix);
    my $distance = $dfun->new($feats->{'train'}, $feats->{'train'}, @$dargs);
    
    my $dm_train=max(abs(
			 $indata->{$prefix.'matrix_train'}
			 -$distance->get_distance_matrix())->flat);
    $distance->init($feats->{'train'}, $feats->{'test'});
    my $dm_test=max(abs(
			$indata->{$prefix.'matrix_test'}
			-$distance->get_distance_matrix())->flat);
    
    return &util::check_accuracy(
	$indata->{$prefix.'accuracy'}
	, {dm_train=>$dm_train, dm_test=>$dm_test});
}


########################################################################
# public
########################################################################

sub test
{
    my ($indata)=@_;
    return &_evaluate($indata);
}

true;
__END__


"""
Test Distance
"""

from shogun.Distance import *

import util


def _evaluate (indata):
	prefix='distance_'
	feats=util.get_features(indata, prefix)

	dfun=eval(indata[prefix+'name'])
	dargs=util.get_args(indata, prefix)
	distance=dfun(feats['train'], feats['train'], *dargs)

	dm_train=max(abs(
		indata[prefix+'matrix_train']-distance.get_distance_matrix()).flat)
	distance.init(feats['train'], feats['test'])
	dm_test=max(abs(
		indata[prefix+'matrix_test']-distance.get_distance_matrix()).flat)

	return util.check_accuracy(
		indata[prefix+'accuracy'], dm_train=dm_train, dm_test=dm_test)


########################################################################
# public
########################################################################

def test (indata):
	return _evaluate(indata)

