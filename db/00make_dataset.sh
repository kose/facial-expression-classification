#!/bin/sh

#
CDIR=$HOME/python/facial-expression-classification/db


tail -312 $CDIR/train_master.tsv | sort -R > /tmp/tmp.tsv

cat<<EOF > $CDIR/train.tsv
id	userid	pose	expression	eyes
EOF
head -290 /tmp/tmp.tsv >> $CDIR/train.tsv


cat<<EOF > $CDIR/test.tsv
id	userid	pose	expression	eyes
EOF
tail -22 /tmp/tmp.tsv >> $CDIR/test.tsv

# end
