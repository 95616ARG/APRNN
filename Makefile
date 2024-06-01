# SHELL=/bin/bash

#     Models
# ==============

verify-models: verify-mnist-models verify-acasxu-models

sha256_mnist_relu_3_100=e4151dfced1783360ab8353c8fdedbfd76f712c2c56e4b14799b2f989217229f
sha256_mnist_relu_9_100=3d84b09ac26b2b174cc51cf2c08f5fc820d000911553b9a2a7653b91efad67d9
sha256_mnist_relu_9_200=3e48e540f83daae615f504c1d92b374d4884bd59418a15cf5b6b970b7265fc4b

verify-mnist-models: \
models/mnist/mnist_relu_3_100.tf \
models/mnist/mnist_relu_9_100.tf \
models/mnist/mnist_relu_9_200.tf
	@bash external/verify_sha256.sh models/mnist/mnist_relu_3_100.tf ${sha256_mnist_relu_3_100}
	@bash external/verify_sha256.sh models/mnist/mnist_relu_9_100.tf ${sha256_mnist_relu_9_100}
	@bash external/verify_sha256.sh models/mnist/mnist_relu_9_200.tf ${sha256_mnist_relu_9_200}

sha256_ACASXU_run2a_2_9_batch_2000=9d07e4b4434c35c466c900ac3138e04b83f7d11bdff4e284b33b4c65674cca1a

verify-acasxu-models: \
models/acasxu/ACASXU_run2a_2_9_batch_2000.nnet
	@bash external/verify_sha256.sh models/acasxu/ACASXU_run2a_2_9_batch_2000.nnet ${sha256_ACASXU_run2a_2_9_batch_2000}


#    Datasets
# ==============

url_mnist_c=https://zenodo.org/record/3239543/files/mnist_c.zip
sha256_mnist_c=af9ee8c6a815870c7fdde5af84c7bf8db0bcfa1f41056db83871037fba70e493
data/mnist_c.zip:
	mkdir -p $(shell dirname $@)
	wget ${url_mnist_c} -P $(shell dirname $@) \
		&& echo "${sha256_mnist_c} $@" | sha256sum --check \
		&& touch $@ \
	|| ( \
		echo "The downloaded $@ does not match the known sha256 ${sha256_mnist_c}." && \
		rm -f $@ \
	)

data/mnist_c: data/mnist_c.zip
	rm -rf $@
	unzip $< -d $(shell dirname $@)
	touch $@

data/ILSVRC2012: data/ILSVRC2012/ILSVRC2012_devkit_t12.tar.gz data/ILSVRC2012/ILSVRC2012_img_val.tar

url_ILSVRC2012_devkit_t12=https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
sha256_ILSVRC2012_devkit_t12=b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953
data/ILSVRC2012/ILSVRC2012_devkit_t12.tar.gz:
	mkdir -p $(shell dirname $@)
	wget ${url_ILSVRC2012_devkit_t12} -P $(shell dirname $@) \
		&& echo "${sha256_ILSVRC2012_devkit_t12} $@" | sha256sum --check \
		&& touch $@ \
	|| ( \
		echo "The downloaded $@ does not match the known sha256 ${sha256_ILSVRC2012_devkit_t12}." && \
		rm -f $@ \
	)

url_ILSVRC2012_img_val=http://academictorrents.com/download/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5.torrent
data/ILSVRC2012/ILSVRC2012_img_val.tar:
	@if test -f $@; then \
		echo "Found \"$@\"."; \
	else \
		echo "Please download the ImageNet validation set \"$@\" (6.3G) from \"${url_ILSVRC2012_img_val}\"."; \
		exit 1; \
	fi

sha256_imagenet-a=3bb3632277e6ba6392ea64c02ddbf4dd2266c9caffd6bc09c9656d28f012589e

data/imagenet-a.tar:
	mkdir -p $(shell dirname $@)
	wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar -P $(shell dirname $@) \
		&& echo "${sha256_imagenet-a} $@" | sha256sum --check \
		&& touch $@ \
	|| ( \
		echo "The downloaded $@ does not match the known sha256 ${sha256_imagenet}." && \
		rm -f $@ \
	)

data/imagenet-a: data/imagenet-a.tar
	rm -rf $@
	tar -xf $< -C $(shell dirname $@)
	touch $@

datasets-mnist: data/mnist_c
datasets-imagenet: data/ILSVRC2012 data/imagenet-a

external/python:
	@mkdir -p $@

clean_python:
	-@ rm -rf external/python
	-@ rm -rf external/python_venv

python_version=3.9.7
python_hash=a838d3f9360d157040142b715db34f0218e535333696a5569dc6f854604eb9d1
external/python/${python_version}.tgz:
	rm -f $@ && \
	mkdir -p $(shell dirname $@) && \
	wget -O $@ https://www.python.org/ftp/python/${python_version}/Python-${python_version}.tgz && \
	echo "${python_hash} $@" | sha256sum --check &&	\
	touch $@ \
	|| ( \
		echo "The downloaded $@ does not match the known sha256 ${python_hash}." && \
		rm -f $@ && \
		exit 1 \
	)

# --enable-optimizations
external/python/${python_version}: external/python/${python_version}.tgz
	rm -rf $(shell dirname $@)/Python-${python_version} && \
	rm -rf $@ && \
	mkdir -p $@ && \
	touch $< && \
	tar -xzf $< -C $(shell dirname $@) && \
	( \
		cd ${PWD}/$(shell dirname $@)/Python-${python_version} && \
		CFLAGS=-fPIC \
			./configure \
				--prefix=${PWD}/$@ \
				--enable-loadable-sqlite-extensions \
				--enable-shared=no && \
		${MAKE} && \
		${MAKE} install && \
		cd - \
	) || ( \
		echo "Failed to build local python $@" && \
		rm -rf $(shell dirname $@) && \
		exit 1 \
	) && \
	touch $@ && \
	rm -rf $(shell dirname $@)/Python-${python_version}

PIP_CACHE_DIR=.pip_cache
external/python_venv/${python_version}: external/python/${python_version} requirements.txt
	rm -rf $@ && \
	$</bin/python3 -m venv $@ && \
	touch requirements.txt && \
	mkdir -p ${PIP_CACHE_DIR} && \
	( \
		. $@/bin/activate && \
		PATH=${PWD}/$</bin:${PWD}/$</include:${PWD}/$</lib:${PATH} && \
		PATH=${PWD}/$@/bin:${PWD}/$@/include:${PWD}/$@/lib:${PWD}/$@/lib64:${PATH} && \
		LD_LIBRARY_PATH=${PWD}/$@/lib:${PWD}/$@/lib64:${PWD}/$</lib:${LD_LIBRARY_PATH} && \
		PYTHONPATH=${PWD}/$@ && \
		python3 -m pip install --upgrade pip && \
		TMPDIR=${PIP_CACHE_DIR} \
			pip3 install -r requirements.txt \
				--upgrade \
				--extra-index-url https://download.pytorch.org/whl \
	) || ( \
		echo "Failed to build local python venv $@" && \
		rm -rf $(shell dirname $@) && \
		exit 1 \
	) && \
	touch $@

venv: external/python_venv/${python_version}

