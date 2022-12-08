set -e 
set -x

cd ../../

# 后续添加拷贝backend到model文件夹

echo "Step1: Install Conda......"
if [ ! -f https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh ];then 
    wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
fi

source /root/miniconda3/bin/activate

echo "Step2: Install Cmake......"
if [ ! -f cmake-3.23.0-rc3-linux-x86_64.tar.gz ];then 
    wget -c https://cmake.org/files/v3.23/cmake-3.23.0-rc3-linux-x86_64.tar.gz
    tar -zxvf cmake-3.23.0-rc3-linux-x86_64.tar.gz
    cp cmake-3.23.0-rc3-linux-x86_64 /opt/cmake-3.23.0 -r
    ln -sf /opt/cmake-3.23.0/bin/* /usr/bin
fi

echo "Step3: Install rapidjson and libarchive......"
apt update
apt-get install rapidjson-dev libarchive-dev zlib1g-dev -y

echo "Step4: Create Python3.7 Env......"
conda create -n python37_mm0.13.0 python=3.7

conda activate python37_mm0.13.0
export PYTHONNOUSERSITE=True
conda install numpy
pip install opencv-python -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple

echo "Step5: Install MagicMind0.13......"
if [ ! -f magicmind-0.13.0-cp37-cp37m-linux_x86_64.whl ];then 
    wget -c http://daily.software.cambricon.com/release/magicmind/Linux/x86_64/Ubuntu/20.04/0.13.0-1/abiold/magicmind-0.13.0-cp37-cp37m-linux_x86_64.whl
    pip install ./magicmind-0.13.0-cp37-cp37m-linux_x86_64.whl
fi 

echo "Step6: Build Triton python backend......"
git clone https://github.com/triton-inference-server/python_backend -b r22.06
cd python_backend
mkdir build && cd build

# -DTRITON_ENABLE_GPU=OFF 应该关掉
cmake -DTRITON_ENABLE_GPU=OFF -DTRITON_BACKEND_REPO_TAG=r22.06 -DTRITON_COMMON_REPO_TAG=r22.06 -DTRITON_CORE_REPO_TAG=r22.06 -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make triton-python-backend-stub -j   
ldd triton_python_backend_stub

cp triton_python_backend_stub  /opt/tritonserver/triton_prj/bert/triton_codes/mm_models/bert_case

cd ../../
conda install conda-pack -y
conda-pack

echo "Step7: Create Cambricon Env......"
if [ ! -f cntoolkit_3.0.2-1.ubuntu20.04_amd64.deb ];then
    wget -c http://daily.software.cambricon.com/release/cntoolkit/Linux/x86_64/Ubuntu/20.04/3.0.2-1/cntoolkit_3.0.2-1.ubuntu20.04_amd64.deb
    dpkg -i cntoolkit_3.0.2-1.ubuntu20.04_amd64.deb
fi
apt update
apt-get install cntoolkit-cloud 

if [ ! -f cntoolkit_3.0.2-1.ubuntu20.04_amd64.deb ];then
    wget -c http://daily.software.cambricon.com/release/cntoolkit/Linux/x86_64/Ubuntu/20.04/3.0.2-1/cntoolkit_3.0.2-1.ubuntu20.04_amd64.deb
    dpkg -i cntoolkit_3.0.2-1.ubuntu20.04_amd64.deb
fi

if [ ! -f cnnl_1.12.1-1.ubuntu20.04_amd64.deb ];then
    wget -c http://daily.software.cambricon.com/release/cnnl/Linux/x86_64/Ubuntu/20.04/1.12.1-1/cnnl_1.12.1-1.ubuntu20.04_amd64.deb
    apt install ./cnnl_1.12.1-1.ubuntu20.04_amd64.deb 
fi

if [ ! -f cnlight_0.15.2-1.abiold.ubuntu20.04_amd64.deb ];then
    wget -c http://daily.software.cambricon.com/release/cnlight/Linux/x86_64/Ubuntu/20.04/0.15.2-1/cnlight_0.15.2-1.abiold.ubuntu20.04_amd64.deb
    apt install ./cnlight_0.15.2-1.abiold.ubuntu20.04_amd64.deb
fi

if [ ! -f cnnlextra_0.18.0-1.ubuntu20.04_amd64.deb ];then
    wget -c http://daily.software.cambricon.com/release/cnnlextra/Linux/x86_64/Ubuntu/20.04/0.18.0-1/cnnlextra_0.18.0-1.ubuntu20.04_amd64.deb
    apt install ./cnnlextra_0.18.0-1.ubuntu20.04_amd64.deb
fi

apt install /var/cntoolkit-3.0.2/llvm-mm-cxx11-old-abi_1.1.1-1.ubuntu20.04_amd64.deb
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:/usr/local/neuware/lib/llvm-mm/lib:$LD_LIBRARY_PATH