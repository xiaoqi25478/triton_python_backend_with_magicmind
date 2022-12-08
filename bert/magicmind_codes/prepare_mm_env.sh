set -e 
set -x

pip install -r requirements.txt

ROOT_PATH=$PWD
mkdir -p $ROOT_PATH/codes
mkdir -p $ROOT_PATH/weights
mkdir -p $ROOT_PATH/models/pt_model
mkdir -p $ROOT_PATH/output

if [ ! -f codes/v3.1.0.zip ];then
    pushd codes
    wget -c https://github.com/huggingface/transformers/archive/refs/tags/v3.1.0.zip -O v3.1.0.zip
    unzip -o v3.1.0.zip
    pip install ./transformers-3.1.0
    popd
else 
    echo "v3.1.0.zip exist!"
fi

pushd codes
pip install ./transformers-3.1.0
popd

if [ ! -f weights/pytorch_bert_base_cased_squad.tgz ];then
    pushd weights
    wget http://gitlab.software.cambricon.com/neuware/software/solutionsdk/pytorch_bert_base_cased_squad_pretrained/-/blob/master/pytorch_bert_base_cased_squad.tgz
    tar -zvxf pytorch_bert_base_cased_squad.tgz 
    popd 
else 
    echo "v3.1.0.zip exist!"
fi

wget -c https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P data/
