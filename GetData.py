import os
from Bio.PDB import PDBList,PDBParser
import random
train_dir = "train_data"
test_dir="test_data"

def download_random_pdbs(data_dir,num_pdbs):
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    # 初始化 PDBList 对象
    pdbl = PDBList()
    # 随机选择 num_pdbs 个 PDB ID
    all_pdb_ids = pdbl.get_all_entries()
    selected_pdb_ids = random.sample(all_pdb_ids, num_pdbs)
    # 下载选定的 PDB 文件
    for pdb_id in selected_pdb_ids:
        try:
            # 下载文件到临时路径
            temp_path = pdbl.retrieve_pdb_file(pdb_id, pdir=data_dir, file_format='pdb')
            # 构造目标文件名
            target_filename = f"{pdb_id.upper()}.pdb"
            target_path = os.path.join(data_dir, target_filename)
            # 重命名文件
            os.rename(temp_path, target_path)
            print(f"成功下载并重命名为 {target_path}")
            if not available(pdb_id,target_path):
                if os.path.exists(target_path):
                    os.remove(target_path)
                    print(f"文件 {target_path} 已成功删除。")
                else:
                    print(f"文件 {target_path} 不存在。")
        except Exception as e:
            print(f"下载或重命名 {pdb_id} 失败: {e}")

def available(pdb_id,pdb_file):
    parser = PDBParser(QUIET=True)#创建解析对象，并抑制警告信息输出
    structure = parser.get_structure(pdb_id, pdb_file)#解析蛋白质结构到内存
    if structure[0]:
        return True
    else:
        return False
            
download_random_pdbs(train_dir, num_pdbs=200)
download_random_pdbs(test_dir,num_pdbs=10)