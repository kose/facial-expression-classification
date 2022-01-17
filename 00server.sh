#!/bin/sh

#
cd `dirname $0`
CDIR=`pwd`

export LANG=C
export PATH=$HOME/miniforge3/bin:$PATH

##
do_command()
{
    echo "\033[32m$1\033[m"
    $1
}

##
## killer
##
server_kill()
{
    PS=`ps x | grep "[t]ensorboard" | awk '{print $1}'`

    if test x"$PS" != x""; then
	echo "kill tensorboard: $PS"
	kill $PS
    fi
}

if test "$1" = "--kill"; then
    server_kill
    exit 0
fi

##
## start server
##
while true; do

    server_kill

    tensorboard --logdir=logs &
    sleep 180
    clear
done

# end
