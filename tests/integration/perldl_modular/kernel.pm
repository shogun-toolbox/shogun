package kernel;

use modshogun;

=pod

comming from modshogun::?
use Shogun::Features;
use Shogun::Kernel;
use Shogun::Preprocessor;
use Shogun::Distance;
use Shogun::Classifier qw(PluginEstimate);
use Shogun::Distribution qw(HMM BW_NORMAL);

=cut


use PDL;#  qw( array ushort ubyte double);

use util;

########################################################################
# kernel computation
########################################################################

sub _evaluate {
 my ($indata, $prefix) = @_;
 my $feats  = &util::get_features($indata, $prefix);
 my $kfun   = eval('modshogun::' . $indata->{$prefix.'name'}.'Kernel');
 my $kargs  = &util::get_args($indata, $prefix);
 my $kernel = $kfun->new(@$kargs);
 if( defined($indata->{$prefix.'normalizer'})) {
     my $fnorm = eval('modshogun::' . $indata->{$prefix.'normalizer'})->new();
     $kernel->set_normalizer($fnorm);
 }
 $kernel->init($feats->{'train'}, $feats->{'train'});
 my $km_train=max(abs(
		   $indata->{$prefix.'matrix_train'}
		   - $kernel->get_kernel_matrix())->flat());
 $kernel->init($feats->{'train'}, $feats->{'test'});
 my $km_test=max(abs(
		  $indata->{$prefix.'matrix_test'}
		  -$kernel->get_kernel_matrix())->flat);

 return &util::check_accuracy(
     $indata->{$prefix.'accuracy'}
     , {km_train => $km_train, km_test => $km_test});
}

sub _get_subkernels 
{
    my ($indata, $prefix) = @_;
    my %subkernels;
    $prefix.='subkernel';
    my $len_prefix = strlen($prefix);

    # loop through indata (unordered dict) to gather subkernel data
    foreach my $key (keys %$indata) {
	unless($prefix =~ /${key}/) { next;}

	# get subkernel's number
	my $num= substr($key, $len_prefix, 1);
	unless ($num) {
	    warn('Cannot find number for subkernel: "%s"!', $data);
	}
	# get item's name
	my $name=substr($key, $len_prefix + 2);

	# append new item
	$subkernels{$num}{$name} = $indata->{$key};

	# got all necessary information in new structure, now create a kernel
	# object for each subkernel
	while(my ($num, $data) = each(%subkernels)) {
	    my $fun = eval('modshogun::' . $data{'name'}.'Kernel');
	    my $args= &util::get_args($data, '');
	    $subkernels{$num}{'kernel'}=$fun->($args);
	}
    }
    return \%subkernels;
}

sub _evaluate_combined
{
    my ($indata, $prefix) = @_;
    my $kernel = modshogun::CombinedKernel->new();
    my %feats=('train'=> modshogun::CombinedFeatures->new()
	       , 'test' => modshogun::CombinedFeatures->new()
	);

    my %subkernels = (&_get_subkernels($indata, $prefix));
    foreach my $subk (keys %subkernels) {
	$feats_subk=&util::get_features($subk, '');
	push(@{$feats{'train'}}, @{$feats_subk{'train'}});

	push(@{$feats{'test'}}, @{$feats_subk{'test'}});
	$kernel->append_kernel($subk{'kernel'});
    }
    $kernel->init($feats{'train'}, $feats{'train'});
    my $km_train=max(abs($indata->{'kernel_matrix_train'}
			 - $kernel->get_kernel_matrix())->flat);
    my $kernel->init($feats{'train'}, $feats{'test'});
    my $km_test=max(abs(
			$indata->{'kernel_matrix_test'}
			- $kernel->get_kernel_matrix())->flat);
    
    return &util::check_accuracy
	(
	 $indata->{$prefix.'accuracy'}
	 , {
	     km_train=>$km_train
		 , km_test=>$km_test
	 }
	);

}
sub _evaluate_auc
{
    my ($indata, $prefix) = @_;
    my $subk=&_get_subkernels(indata, prefix)->{'0'};
    my $feats_subk = &util::get_features($subk, '');
    $subk{'kernel'}->init($feats_subk{'train'}, $feats_subk{'test'});
    my %feats= (
	'train'=> modshogun::WordFeatures->new($indata->{$prefix.'data_train'}->astype(&ushort))
	, 'test'=> modshogun::WordFeatures->new($indata->{$prefix.'data_test'}->astype(&ushort))
	);
    my $kernel= modshogun::AUCKernel->new(10, $subk{'kernel'});

    $kernel->init($feats{'train'}, $feats{'train'});
    my $km_train=max(abs(
			 $indata->{$prefix.'matrix_train'}-$kernel->get_kernel_matrix())->flat);
    $kernel->init(feats{'train'}, $feats{'test'});
    my $km_test=max(abs(
			$indata->{$prefix.'matrix_test'}-$kernel->get_kernel_matrix())->flat);

    return &util::check_accuracy
	($indata->{$prefix.'accuracy'}
	 , {   km_train=>$km_train
		   , km_test=>$km_test
	 }
	);

}
sub  _evaluate_custom
{
    my ($indata, $prefix) = @_;
    my %feats=(
		'train'=> modshogun::RealFeatures->new($indata->{$prefix.'data'}),
		'test'=> modshogun::RealFeatures->new($indata->{$prefix.'data'})
	    );

    my $symdata=$indata->{$prefix.'symdata'};
    if(0) {
#PTZ121005not realy sure about this...
    my @c;
    foreach my  $x (0..$symdata->shape()->at(1)-1) {
	my @r;
	foreach my $y (0..$symdata->shape()->at(0)-1) {
	    if($y <= $x) {
		push(@r, $symdata->at($x, $y));
	    }
	}
	push(@c, \@r);
    }

    my $lowertriangle = pdl(\@c);
    }
    my $ns = $symdata->shape()->at(0);
    my $lowertriangle = zeroes($ns * ($ns + 1) / 2);
    $symdata->squaretotri($lowertriangle);

#
#           if($SIZE (m) != (ns * (ns+1))/2) {
#               barf("Wrong sized args for squaretotri");
#            }
#            threadloop %{
#                loop(m) %{
#                       $b () = $a (n0 => mna, n1 => nb);
#                      mna++; if(mna > nb) {mna = 0; nb ++;}
#                %}
#
#	lowertriangle=array([symdata[(x,y)] for x in xrange(symdata.shape[1])
#		for y in xrange(symdata.shape[0]) if y<=x])

    my $kernel= modshogun::CustomKernel->new();
    $kernel->set_triangle_kernel_matrix_from_triangle($lowertriangle);
    my $triangletriangle
	=max(abs(
		 $indata->{$prefix.'matrix_triangletriangle'}
		 -$kernel->get_kernel_matrix())->flat);
    $kernel->set_triangle_kernel_matrix_from_full($indata->{$prefix.'symdata'});
    my $fulltriangle
	=max(abs(
		 $indata->{$prefix.'matrix_fulltriangle'}
		 -$kernel->get_kernel_matrix())->flat);
    $kernel->set_full_kernel_matrix_from_full($indata->{$prefix.'data'});
    my $fullfull
	=max(abs(
		 $indata->{$prefix.'matrix_fullfull'}
		 -$kernel->get_kernel_matrix())->flat);

    return &util::check_accuracy
	($indata->{$prefix.'accuracy'}
	 ,{
	     triangletriangle=>$triangletriangle
		 , fulltriangle=>$fulltriangle
		 , fullfull=>$fullfull
	 }
	);
}

sub _evaluate_pie
{
    my ($indata, $prefix) = @_;
    my $pie=modshogun::PluginEstimate->new();
    my $feats=&util::get_features($indata, $prefix);
    my $labels=modshogun::BinaryLabels->new(&double($indata->{'classifier_labels'}));
    $pie->set_labels($labels);
    $pie->set_features($feats->{'train'});
    $pie->train();

    my $fun=*{$indata->{$prefix.'name'}.'Kernel'};
    my $kernel=$fun->($feats->{'train'}, $feats->{'train'}, $pie);
    my $km_train=max(abs(
			 $indata->{$prefix.'matrix_train'}
			 -$kernel->get_kernel_matrix())->flat);

    $kernel->init($feats->{'train'}, $feats->{'test'});
    $pie->set_features(feats->{'test'});
    my $km_test=max(abs(
			$indata->{$prefix.'matrix_test'}
			-$kernel->get_kernel_matrix())->flat);
    my $classified=max(abs(
		$pie->apply()->get_confidences()
			   -$indata->{'classifier_classified'}));
    
    return &util::check_accuracy(
	$indata->{$prefix.'accuracy'}
	, {
	    km_train=>$km_train, km_test=>$km_test, classified=>$classified});
    
}
			    
sub _evaluate_top_fisher
{
    my ($indata, $prefix) = @_;

    my %feats;
    my $wordfeats=&util::get_features($indata, $prefix);

    my $pos_train=&modshogun::HMM($wordfeats{'train'}, $indata->{$prefix.'N'}, $indata->{$prefix.'M'},
		       $indata->{$prefix.'pseudo'});
    $pos_train->train();
    $pos_train->baum_welch_viterbi_train($modshogun::BW_NORMAL);
    my $neg_train= modshogun::HMM->new($wordfeats{'train'}, $indata->{$prefix.'N'}, $indata->{$prefix.'M'},
		       $indata->{$prefix.'pseudo'});
    $neg_train->train();
    $neg_train->baum_welch_viterbi_train($modshogun::BW_NORMAL);
    my $pos_test= modshogun::HMM->new($pos_train);
    $pos_test->set_observations($wordfeats{'test'});
    my $neg_test= modshogun::HMM->new($neg_train);
    $neg_test->set_observations($wordfeats{'test'});

    if($indata->{$prefix.'name'} eq 'TOP'){
	$feats->{'train'}= modshogun::TOPFeatures->new(10, $pos_train, $neg_train, false, false);
	$feats->{'test'}= modshogun::TOPFeatures->new(10, $pos_test, $neg_test, false, false);
    }else{
	$feats->{'train'}= modshogun::FKFeatures->new(10, $pos_train, $neg_train);
	$feats->{'train'}->set_opt_a(-1); #estimate prior
	$feats->{'test'}= modshogun::FKFeatures->new(10, $pos_test, $neg_test);
	$feats->{'test'}->set_a($feats->{'train'}->get_a()); #use prior from training data
    }
    $prefix='kernel_';
    my $args=&util::get_args($indata, $prefix);
    my $kernel= modshogun::PolyKernel->new($feats->{'train'}, $feats->{'train'}, $args);
#	kernel=PolyKernel(*args)
#	kernel.init(feats['train'], feats['train'])
    my $km_train=max(abs(
		$indata->{$prefix.'matrix_train'}
			 -$kernel->get_kernel_matrix())->flat);
    $kernel->init($feats->{'train'}, $feats->{'test'});
    my $km_test=max(abs(
		$indata->{$prefix.'matrix_test'}
			-$kernel->get_kernel_matrix())->flat);

    return &util::check_accuracy($indata->{$prefix.'accuracy'}
			       , {km_train=>$km_train, km_test=>$km_test});

}
########################################################################
# public
########################################################################

sub test
{
    my ($indata) =@_;
    my $prefix='topfk_';
    if(defined($indata->{$prefix.'name'})) {
	return &_evaluate_top_fisher($indata, $prefix);
    }
    $prefix='kernel_';
    my @names=('Combined', 'AUC', 'Custom');
    foreach my $name (@names) {
	if( $indata->{$prefix.'name'}eq $name) {
	    return eval('&_evaluate_'. lc $name . '($indata, $prefix)');
	}
    }
    @names=('HistogramWordString', 'SalzbergWordString');
    foreach my $name (@names) {
	if($indata->{$prefix . 'name'} eq $name) {
	    return &_evaluate_pie($indata, $prefix);
	}
    }
    return &_evaluate($indata, $prefix);
}

true;
__END__
=head1

Test Kernel

=cut

from shogun.Features import *
from shogun.Kernel import *
from shogun.Preprocessor import *
from shogun.Distance import *
from shogun.Classifier import PluginEstimate
from shogun.Distribution import HMM, BW_NORMAL
from numpy import array, ushort, ubyte, double

import util

########################################################################
# kernel computation
########################################################################

def _evaluate (indata, prefix):
	feats=util.get_features(indata, prefix)
	kfun=eval($indata->{$prefix+'name']+'Kernel')
	kargs=util.get_args(indata, prefix)
	kernel=kfun(*kargs)
	if indata.has_key(prefix+'normalizer'):
		kernel.set_normalizer(eval($indata->{$prefix+'normalizer']+'()'))

	kernel.init(feats['train'], feats['train'])
	km_train=max(abs(
		$indata->{$prefix+'matrix_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	km_test=max(abs(
		$indata->{$prefix+'matrix_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy(
		$indata->{$prefix+'accuracy'], km_train=km_train, km_test=km_test)


def _get_subkernels (indata, prefix):
	subkernels={}
	prefix=prefix+'subkernel'
	len_prefix=len(prefix)

	# loop through indata (unordered dict) to gather subkernel data
	for key in indata:
		if key.find(prefix)==-1:
			continue

		# get subkernel's number
		try:
			num=key[len_prefix]
		except ValueError:
			raise ValueError, 'Cannot find number for subkernel: "%s"!' % data

		# get item's name
		name=key[len_prefix+2:]

		# append new item
		if not subkernels.has_key(num):
			subkernels[num]={}
		subkernels[num][name]=indata[key]

	# got all necessary information in new structure, now create a kernel
	# object for each subkernel
	for num, data in subkernels.iteritems():
		fun=eval(data['name']+'Kernel')
		args=util.get_args(data, '')
		subkernels[num]['kernel']=fun(*args)

	return subkernels


def _evaluate_combined (indata, prefix):
	kernel=CombinedKernel()
	feats={'train':CombinedFeatures(), 'test':CombinedFeatures()}

	subkernels=_get_subkernels(indata, prefix)
	for subk in subkernels.itervalues():
		feats_subk=util.get_features(subk, '')
		feats['train'].append_feature_obj(feats_subk['train'])
		feats['test'].append_feature_obj(feats_subk['test'])
		kernel.append_kernel(subk['kernel'])

	kernel.init(feats['train'], feats['train'])
	km_train=max(abs(
		indata['kernel_matrix_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	km_test=max(abs(
		indata['kernel_matrix_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy($indata->{$prefix+'accuracy'],
		km_train=km_train, km_test=km_test)


def _evaluate_auc (indata, prefix):
	subk=_get_subkernels(indata, prefix)['0']
	feats_subk=util.get_features(subk, '')
	subk['kernel'].init(feats_subk['train'], feats_subk['test'])

	feats={
		'train': WordFeatures($indata->{$prefix+'data_train'].astype(ushort)),
		'test': WordFeatures($indata->{$prefix+'data_test'].astype(ushort))
	}
	kernel=AUCKernel(10, subk['kernel'])

	kernel.init(feats['train'], feats['train'])
	km_train=max(abs(
		$indata{$prefix+'matrix_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	km_test=max(abs(
		$indata{$prefix+'matrix_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy($indata{$prefix+'accuracy'],
		km_train=km_train, km_test=km_test)


def _evaluate_custom (indata, prefix):
	feats={
		'train': RealFeatures($indata{$prefix+'data']),
		'test': RealFeatures($indata{$prefix+'data'])
	}

	symdata=$indata{$prefix+'symdata']
	lowertriangle=array([symdata[(x,y)] for x in xrange(symdata.shape[1])
		for y in xrange(symdata.shape[0]) if y<=x])
	kernel=CustomKernel()
	kernel.set_triangle_kernel_matrix_from_triangle(lowertriangle)
	triangletriangle=max(abs(
		$indata{$prefix+'matrix_triangletriangle']-kernel.get_kernel_matrix()).flat)
	kernel.set_triangle_kernel_matrix_from_full($indata{$prefix+'symdata'])
	fulltriangle=max(abs(
		$indata{$prefix+'matrix_fulltriangle']-kernel.get_kernel_matrix()).flat)
	kernel.set_full_kernel_matrix_from_full($indata{$prefix+'data'])
	fullfull=max(abs(
		$indata{$prefix+'matrix_fullfull']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy($indata{$prefix+'accuracy'],
		triangletriangle=triangletriangle, fulltriangle=fulltriangle,
		fullfull=fullfull)


def _evaluate_pie (indata, prefix):
	pie=PluginEstimate()
	feats=util.get_features(indata, prefix)
	labels=BinaryLabels(double(indata['classifier_labels']))
	pie.set_labels(labels)
	pie.set_features(feats['train'])
	pie.train()

	fun=eval($indata{$prefix+'name']+'Kernel')
	kernel=fun(feats['train'], feats['train'], pie)
	km_train=max(abs(
		$indata{$prefix+'matrix_train']-kernel.get_kernel_matrix()).flat)

	kernel.init(feats['train'], feats['test'])
	pie.set_features(feats['test'])
	km_test=max(abs(
		$indata{$prefix+'matrix_test']-kernel.get_kernel_matrix()).flat)
	classified=max(abs(
		pie.apply().get_confidences()-indata['classifier_classified']))

	return util.check_accuracy($indata{$prefix+'accuracy'],
		km_train=km_train, km_test=km_test, classified=classified)


def _evaluate_top_fisher (indata, prefix):
	feats={}
	wordfeats=util.get_features(indata, prefix)

	pos_train=HMM(wordfeats['train'], $indata{$prefix+'N'], $indata{$prefix+'M'],
		$indata{$prefix+'pseudo'])
	pos_train.train()
	pos_train.baum_welch_viterbi_train(BW_NORMAL)
	neg_train=HMM(wordfeats['train'], $indata{$prefix+'N'], $indata{$prefix+'M'],
		$indata{$prefix+'pseudo'])
	neg_train.train()
	neg_train.baum_welch_viterbi_train(BW_NORMAL)
	pos_test=HMM(pos_train)
	pos_test.set_observations(wordfeats['test'])
	neg_test=HMM(neg_train)
	neg_test.set_observations(wordfeats['test'])

	if $indata{$prefix+'name']=='TOP':
		feats['train']=TOPFeatures(10, pos_train, neg_train, False, False)
		feats['test']=TOPFeatures(10, pos_test, neg_test, False, False)
	else:
		feats['train']=FKFeatures(10, pos_train, neg_train)
		feats['train'].set_opt_a(-1) #estimate prior
		feats['test']=FKFeatures(10, pos_test, neg_test)
		feats['test'].set_a(feats['train'].get_a()) #use prior from training data

	prefix='kernel_'
	args=util.get_args(indata, prefix)
	kernel=PolyKernel(feats['train'], feats['train'], *args)
#	kernel=PolyKernel(*args)
#	kernel.init(feats['train'], feats['train'])
	km_train=max(abs(
		$indata{$prefix+'matrix_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	km_test=max(abs(
		$indata{$prefix+'matrix_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy($indata{$prefix+'accuracy'],
		km_train=km_train, km_test=km_test)


########################################################################
# public
########################################################################

def test (indata) = @_; 
	prefix='topfk_'
	if indata.has_key(prefix+'name'):
		return _evaluate_top_fisher(indata, prefix)

	prefix='kernel_'
	names=['Combined', 'AUC', 'Custom']
	for name in names:
		if $indata{$prefix.'name']==name:
			return eval('_evaluate_'+name.lower()+'(indata, prefix)')

	names=['HistogramWordString', 'SalzbergWordString']
	for name in names:
		if $indata{$prefix . 'name']==name:
			return _evaluate_pie(indata, prefix)

	return _evaluate(indata, prefix)

