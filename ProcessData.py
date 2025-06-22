import os
from sklearn.preprocessing import LabelEncoder
from Bio.PDB import PDBParser, DSSP
from torch.nn.utils.rnn import pad_sequence
import torch

# 定义目录路径
train_dir = "train_data"
test_dir = "test_data"

#氨基酸数字化
standard_amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
aa_encoder = LabelEncoder().fit(standard_amino_acids)

#二级结构数字化
# H -- α螺旋
# E -- β折叠
# C -- 无规卷曲
ss_encoder = LabelEncoder().fit(['H', 'E', 'C']) 

def process(pdb_id,id_list,file_list):
    index=id_list.index(pdb_id)
    pdb_file=file_list[index]
    try:
        # 解析文件
        parser = PDBParser(QUIET=True)#创建解析对象，并抑制警告信息输出
        structure = parser.get_structure(pdb_id, pdb_file)#解析蛋白质结构到内存
        dssp = DSSP(structure[0], pdb_file)#计算二级结构
        #dssp结构残基 ('A', (' ', 1, ' ')): ('H', 115.2, -2.7, -0.5, -1.2, -57.8, -47.3, 0.0, 0.0, '-')
        #简化+氨基酸过滤
        seq, ss = [], []
        #seq氨基酸序列 ss二级结构序列
        for residue in dssp:
            # aa是氨基酸类型 ss是二级结构
            aa = residue[1].upper()
            ss_code = residue[2]
            # 简化二级结构标签
            simplified_ss = 'H' if ss_code in ['H','G','I'] else \
                           'E' if ss_code in ['E','B'] else 'C'
            
            if aa in standard_amino_acids:
                seq.append(aa)
                ss.append(simplified_ss)
        
        return seq, ss
    except Exception as e:
        print(f"处理错误 {pdb_id}: {e}")
        return None, None

#///////////////////////////////////////////////////////////

def predict(model,pdb_id,pdb_ids_test,pdb_files_test):
    seq, ss = process(pdb_id, pdb_ids_test, pdb_files_test)
    if not seq:
        return
    
    print(f"序列长度: {len(seq)}")
    print(f"二级结构长度: {len(ss)}")
    
    if len(seq) != len(ss):
        print("序列长度与二级结构长度不一致，无法进行预测。")
        return
    
    # 编码并转换为张量
    x = torch.tensor(aa_encoder.transform(seq), dtype=torch.long).unsqueeze(0)
    
    # 预测
    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(x), dim=2)[0]
    
    # 确保预测长度与真实长度一致
    pred = pred[:len(seq)]
    
    # 解码结果
    true_ss = ''.join(ss_encoder.inverse_transform(ss_encoder.transform(ss)))
    pred_ss = ''.join(ss_encoder.inverse_transform(pred.numpy()))
    
    print(f"\n蛋白质 {pdb_id} 预测结果:")
    print("真实二级结构:", true_ss)
    print("预测二级结构:", pred_ss)
    print("准确率:", (pred == torch.tensor(ss_encoder.transform(ss))).float().mean().item())

#////////////////////////////////////////////

def read():
    # 初始化列表
    pdb_ids_train = []
    pdb_files_train = []
    pdb_ids_test = []
    pdb_files_test = []
    # 读取训练数据目录
    for filename in os.listdir(train_dir):
        if filename.endswith(".pdb"):
            pdb_id = filename[:-4]  # 去掉文件扩展名
            pdb_ids_train.append(pdb_id)
            pdb_files_train.append(os.path.join(train_dir, filename))

    # 读取测试数据目录
    for filename in os.listdir(test_dir):
        if filename.endswith(".pdb"):
            pdb_id = filename[:-4]  # 去掉文件扩展名
            pdb_ids_test.append(pdb_id)
            pdb_files_test.append(os.path.join(test_dir, filename))

    # 打印结果
    print("Training PDB IDs:", pdb_ids_train)
    print("Training PDB Files:", pdb_files_train)
    print("Testing PDB IDs:", pdb_ids_test)
    print("Testing PDB Files:", pdb_files_test)
    
    return pdb_ids_train,pdb_files_train,pdb_ids_test,pdb_files_test

def Pretrain(pdb_ids_train,pdb_files_train):
    X_list, y_list = [], []
    for pdb_id in pdb_ids_train:
        seq, ss = process(pdb_id,pdb_ids_train,pdb_files_train)
        if seq:
            X_list.append(torch.tensor(aa_encoder.transform(seq), dtype=torch.long))
            y_list.append(torch.tensor(ss_encoder.transform(ss), dtype=torch.long))
            print(f"已加载 {pdb_id} (长度: {len(seq)})")

    if not X_list:
        raise ValueError("没有有效数据！请检查PDB ID或网络连接")

    # 填充序列
    X = pad_sequence(X_list, batch_first=True, padding_value=0)
    y = pad_sequence(y_list, batch_first=True, padding_value=-1)
    print(f"\n最终数据形状: X={X.shape}, y={y.shape}")
    
    return X,y

def save(X,y):
    torch.save(X, 'X.pt')
    torch.save(y, 'y.pt')
    print("数据已成功保存到磁盘。")

if __name__ == '__main__':
    pdb_ids_train,pdb_files_train,_1,_2=read()
    X,y=Pretrain(pdb_ids_train,pdb_files_train)
    save(X,y)