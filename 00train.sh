#!/bin/sh

#
cd `dirname $0`
CDIR=`pwd`

export LANG=C


##
do_command()
{
    echo "\033[32m$1\033[m"
    $1
}

##
## learn  func
##

learn()
{
    PREFIX=$1
    DIM1=$2
    DIM2=$3
    DIM3=$4
    EPOCH=$5
    
    LOG="$PREFIX""$DIM1"_"$DIM2"_"$DIM3"-`date +%Y%m%d_%H%M%S`

    if test -f result/$LOG.pt; then
	echo "exist result/$LOG.pt"
    else

	#
	# sh db/00make_dataset.sh
	# cp db/train.tsv db/"$LOG"_train.tsv
	# cp db/test.tsv db/"$LOG"_test.tsv

	do_command "time python train.py --dim1=$DIM1 --dim2=$DIM2 --dim3=$DIM3 --log $LOG --num-processes 6 --epoch $EPOCH"
	do_command "python test.py --model result/$LOG.pt --csv result/$LOG.csv"
	do_command "cp result/$LOG.pt result/result.pt"
    fi
}

##
## learn
##

PREFIX="dim-"

learn "$PREFIX" 16 32 32 100
learn "$PREFIX" 32 64 64 100
learn "$PREFIX" 64 128 128 100
learn "$PREFIX" 128 256 256 100

# end
