package clustering;
use util;


sub _evaluate
{
    my ($indata) = @_;
    my $first_arg;
    if(defined( $indata->{'clustering_k'})) {
	$first_arg=$indata->{'clustering_k'};
    }elsif(defined($indata->{'clustering_merges'})) {
	$first_arg=$indata->{'clustering_merges'};
    }else{
	return false;
    }
    my $feats=&util::get_features($indata, 'distance_');
    my $dfun=*{$indata->{'distance_name'}};
    my $distance=$dfun->($feats->{'train'}, $feats->{'train'});

    my $cfun=*{$indata->{'clustering_name'}};
    my $clustering=$cfun->($first_arg, $distance);
    $clustering->train();

    if(defined($indata->{'clustering_radi'})) {
	my $radi=max(abs($clustering->get_radiuses()-$indata->{'clustering_radi'}));
	my $centers=max(abs($clustering->get_cluster_centers()->flatten()
			    - $indata->{'clustering_centers'}->flat()));
	return(&util::check_accuracy(
		    $indata->{'clustering_accuracy'},
		    {radi=>$radi, centers=>$centers}));
    } elsif(defined( $indata->{'clustering_merge_distance'})) {
	my $merge_distance=max(abs($clustering->get_merge_distances()
				   - $indata->{'clustering_merge_distance'}));
	$pairs=max(abs($clustering->get_cluster_pairs()
		       - $indata->{'clustering_pairs'}->flat()));
	return &util::check_accuracy($indata->{'clustering_accuracy'}
				     , {merge_distance=>$merge_distance, pairs=>$pairs}
	    );
    } else {
	return &util::check_accuracy($indata->{'clustering_accuracy'});
    }
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
Test Clustering
"""

from shogun.Distance import EuclideanDistance
from shogun.Clustering import *

import util


def _evaluate (indata):
	if indata.has_key('clustering_k'):
		first_arg=indata['clustering_k']
	elif indata.has_key('clustering_merges'):
		first_arg=indata['clustering_merges']
	else:
		return False

	feats=util.get_features(indata, 'distance_')
	dfun=eval(indata['distance_name'])
	distance=dfun(feats['train'], feats['train'])

	cfun=eval(indata['clustering_name'])
	clustering=cfun(first_arg, distance)
	clustering.train()

	if indata.has_key('clustering_radi'):
		radi=max(abs(clustering.get_radiuses()-indata['clustering_radi']))
		centers=max(abs(clustering.get_cluster_centers().flatten() - \
			indata['clustering_centers'].flat))
		return util.check_accuracy(indata['clustering_accuracy'],
			radi=radi, centers=centers)
	elif indata.has_key('clustering_merge_distance'):
		merge_distance=max(abs(clustering.get_merge_distances()- \
			indata['clustering_merge_distance']))
		pairs=max(abs(clustering.get_cluster_pairs()- \
			indata['clustering_pairs']).flat)
		return util.check_accuracy(indata['clustering_accuracy'],
			merge_distance=merge_distance, pairs=pairs)
	else:
		return util.check_accuracy(indata['clustering_accuracy'])


########################################################################
# public
########################################################################

def test (indata):
	return _evaluate(indata)

