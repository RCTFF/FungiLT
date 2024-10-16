import sys
from models.utils import *
from utils import *
from feature_extraction.sequence_encoding import *
from feature_extraction.labels_encoding import *
from models.cnn_model import *
from train.train import *
from train.test import *
from data.data import *
import time
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# 并行计算步骤 1/2 设置环境变量
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def main():
    
    # 记录当前时间
    current_time = read_current_time(time.time())
    print(f"==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== =====")
    print(f"==== {current_time} 程序开始执行 ==== ==== ====")
    
    # 获取配置文件
    config = load_config()
    USING_PARALLEL_COMPUTING = config['parallel_computing']['enable']
    MASK_DATA_AUGMENTATION = config['mask_data_augmentation']['enable']
    TRAIN_MODE = config['train_mode']['enable'] 
    TEST_MODE = config['test_mode']['enable']
    KMERS_FREQ_X = 1 if config['seq_encoder']['name'] == 'kmer_freq_X' else 0
    # 交叉验证开启标志
    KFOLD_ENABLE = 1 if config['kfold_cross_validation']['enable'] == 1 else 0
    # 三视角
    THREE_DATA = 1 if config['three_data']['enable'] == 1 else 0
    
    
    # 设置随机种子
    set_seed()
    
    # 数据处理步骤 1/ 获取文件路径
   
    input_file = select_file_path()
    if KMERS_FREQ_X == 1:
        # 读取kmers的进阶版
        seqs_kmer, labels = read_seqkmers_and_labels_todict2(input_file, config["k_mer"]["k"])
    else:
        # 数据处理步骤 2/ 获取数据，kmer编码
        seqs_kmer, labels = read_seqkmers_and_labels_todict(input_file, config["k_mer"]["k"])
            
    # print(f"kmer序列示例:{seqs_kmer[0]}")
    # 数据处理步骤 3/ 对kmer序列编码
    if THREE_DATA == 0:
        data1, data2 = select_sequences_encoder(seqs_kmer)
    else:
        data1, data2, data3 = select_sequences_encoder2(seqs_kmer)
            
    # 数据处理步骤 4/ 对标签数据one-hot编码 （data,labels现在是numpy数组格式）
    labels = select_labels_encoder(labels)
    # labels = create_one_hot_encoding(labels)
    # labels = convert_to_int_labels(labels)
    
    # 获取分类数,数据长度   
    # class_num = len(labels[0])
    class_num = len(data1) 
    data_len = len(data1[0])
    print(data1[0].shape)
    # print(data1[0])
    
    # 选择使用哪个模型
    model, conv_dim = select_model(class_num)
    
    # 并行计算步骤 2/2 模型并行设置
    device_ids = [0,1]
    if USING_PARALLEL_COMPUTING == 1:   # 是否开启多卡并行计算
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.to('cuda')
    else:
        model = model.to('cuda')
    
    # 数据处理步骤 5/ 装载并分割数据集
    if THREE_DATA == 0:
        if KFOLD_ENABLE == 1:   # 是否开启交叉验证
            if data2 is None:
                train_dataset, test_dataset = get_mydataset_kfold(data1, labels, conv_dim)
                if TRAIN_MODE == 1:         # 是否开启训练模式
                    train_kfold(model, train_dataset)
            else:
                # 1 两组输入数据并行
                train_dataset, test_dataset = get_mydataset_kfold2(data1, data2, labels, conv_dim)
                if TRAIN_MODE == 1:
                    train_kfold2(model, train_dataset)
        else:
            if data2 is None:
                train_dataset, val_dataset, test_dataset = get_mydataset(data1, labels, conv_dim)
                if TRAIN_MODE == 1:         # 是否开启训练模式
                    train(model, train_dataset, val_dataset)
            else:
                # 1 两组输入数据并行
                train_dataset, val_dataset, test_dataset= get_mydataset1(data1, data2, labels, conv_dim)
                if TRAIN_MODE == 1:
                    train1(model, train_dataset, val_dataset)
        
        if TEST_MODE == 1:              # 是否开启测试模式
            if data2 is None:
                # 测试
                test(test_dataset)
                # test(val_dataset)
            else:
                test1(test_dataset)
    else:
        train_dataset, val_dataset, test_dataset= get_mydataset3(data1, data2, data3, labels, conv_dim)
        if TRAIN_MODE == 1:
            train3(model, train_dataset, val_dataset)
        if TEST_MODE == 1:              # 是否开启测试模式
            test3(test_dataset)
    
    current_time = read_current_time(time.time())
    print(f"==== 程序执行完毕，当前时间：{current_time}{'':<7} ==== ==== ====")
    
    
if __name__ == "__main__" :
    
    main()
