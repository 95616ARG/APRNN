#!/usr/bin/env bash

echo "$2 $1" | sha256sum --check \
|| ( \
    echo "\"$@\" does not match the known sha256 \"${sha256_mnist_c}\"." && \
    exit 1 \
)
