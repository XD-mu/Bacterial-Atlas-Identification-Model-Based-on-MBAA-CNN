import os
import glob
import shutil
import random

origin_data_folder = './Origin_Data/data2'
final_data_folder = './Final_Data/data2'

bacteria_folders = glob.glob(os.path.join(origin_data_folder, '*'))

for bacteria_folder in bacteria_folders:
    # 提取细菌名称
    bacteria_name = os.path.basename(bacteria_folder)

    # 获取该细菌种类文件夹下所有txt文件的路径
    txt_files = glob.glob(os.path.join(bacteria_folder, '*.txt'))

    # 打乱txt_files的顺序
    random.shuffle(txt_files)

    # 计算训练集和测试集的分割点
    num_files = len(txt_files)
    train_ratio = 0.8  
    test_ratio = 0.2   
    num_train_files = int(num_files * train_ratio)
    num_test_files = int(num_files * test_ratio)
    num_test_files_to_train = int(num_test_files * 0.97)

    # 创建训练集和测试集文件夹
    train_folder = os.path.join(final_data_folder, 'train')
    test_folder = os.path.join(final_data_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 循环处理每个txt文件
    for i, txt_file in enumerate(txt_files):
        # 根据实验次数和文件名构建新的文件名
        experiment_num = i + 1
        new_file_name = f'{bacteria_name}_experiment_{experiment_num}.txt'

        # 打开原始文件和目标文件
        with open(txt_file, 'r') as input_file, open(new_file_name, 'w') as output_file:
            # 读取原始文件的所有行
            lines = input_file.readlines()

            # 删除第一行，保留从第二行开始的数据
            lines = lines[1:]

            # 处理每一行数据
            for line in lines:
                # 切分数据，使用Tab作为分隔符
                columns = line.strip().split('\t')

                # 只保留后两列数据
                if len(columns) >= 2:
                    new_line = '\t'.join(columns[-2:]) + '\n'
                    output_file.write(new_line)
        if i < num_train_files:
            shutil.move(new_file_name, train_folder)
        else:
            shutil.copy(new_file_name, train_folder) if i < num_train_files + num_test_files_to_train else None
            shutil.move(new_file_name, test_folder)
        print(f'Successfully processed {txt_file} to {new_file_name}')

print('Data preprocessing and file allocation completed.')
