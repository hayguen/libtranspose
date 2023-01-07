#!/bin/bash

if [ -z "${VERBOSE}" ]; then
    VERBOSE=""    # "-v" or "" (empty)
fi
if [ -z "${W_A}" ]; then
    W_A="304"     # unaligned: 303   aligned: 304
fi
if [ -z "${H_A}" ]; then
    H_A="1024"    # unaligned: 1023  aligned: 1024
fi
if [ -z "${W_B}" ]; then
    W_B="512"     # unaligned: 511   aligned: 512
fi
if [ -z "${H_B}" ]; then
    H_B="2048"    # unaligned: 2047  aligned: 2048
fi

if [ -z "${PLOTMS}" ]; then
    PLOTMS="8" # plot 8 MSamples data
fi

if [ -z "${ITERSA}" ]; then
    ITERSA="100" # iterations for 300 x 1024 or 1024 x 300; >1 to allow readable millis; =1 to get raw cycles
fi
if [ -z "${ITERSB}" ]; then
    ITERSB="25"  # iterations for 500 x 2048 or 2048 x 500;
fi


if [ -z "$1" ]; then
    BENCHVARIANT="bench44"
else
    BENCHVARIANT="bench$1"
    shift
fi

if [ -x build/${BENCHVARIANT} ]; then
    BENCH="build/${BENCHVARIANT}"
elif [ -x $1/${BENCHVARIANT} ]; then
    BENCH="$1/${BENCHVARIANT}"
else
    echo "Error could not find built 'bench'"
    exit 1
fi

echo "will use executable ${BENCH}"


# taskset to force CPU affinity for benchmarking
#   see https://linux.die.net/man/1/taskset

SUCCESS=""
echo -e "\n=== ${H_A} x ${W_A} ===\n"
if taskset 1 ${BENCH} ${VERBOSE} ${ITERSA} ${PLOTMS} ${H_A} ${W_A} ; then
    echo -e "\n=== ${W_A} x ${H_A} ===\n"
    if taskset 1 ${BENCH} ${VERBOSE} ${ITERSA} ${PLOTMS} ${W_A} ${H_A} ; then
        if [ "${W_B}" = "0" ] || [ "${H_B}" = "0" ]; then
             SUCCESS="1"
        else
            echo -e "\n=== ${H_B} x ${W_B} ===\n"
            if taskset 1 ${BENCH} ${VERBOSE} ${ITERSB} ${PLOTMS} ${H_B} ${W_B} ; then
                echo -e "\n=== ${W_B} x ${H_B} ===\n"
                if taskset 1 ${BENCH} ${VERBOSE} ${ITERSB} ${PLOTMS} ${W_B} ${H_B} ; then
                    SUCCESS="1"
                fi
            fi
        fi
    fi
fi

if [ -z "${SUCCESS}" ]; then
    echo -e "\n=== error ===\n"
else
    echo -e "\n=== success ===\n"
fi
