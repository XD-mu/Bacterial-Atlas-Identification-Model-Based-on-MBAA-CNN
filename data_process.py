# import os
# import glob
# import shutil
# import random
# import pandas as pd
#
# # 设置原始数据文件夹和最终数据文件夹路径
# origin_data_folder = './Origin_Data'
# final_data_folder = './Final_Data'
#
# # 获取所有细菌种类文件夹的路径
# bacteria_folders = glob.glob(os.path.join(origin_data_folder, '*'))
#
# # 循环处理每个细菌种类文件夹
# for bacteria_folder in bacteria_folders:
#     # 提取细菌名称
#     bacteria_name = os.path.basename(bacteria_folder)
#
#     # 获取该细菌种类文件夹下所有xlsx文件的路径
#     xlsx_files = glob.glob(os.path.join(bacteria_folder, '*.xlsx'))
#
#     # 打乱xlsx_files的顺序
#     random.shuffle(xlsx_files)
#
#     # 计算训练集和测试集的分割点
#     num_files = len(xlsx_files)
#     train_ratio = 0.8
#     num_train_files = int(num_files * train_ratio)
#
#     # 创建训练集和测试集文件夹
#     train_folder = os.path.join(final_data_folder, 'train')
#     test_folder = os.path.join(final_data_folder, 'test')
#     os.makedirs(train_folder, exist_ok=True)
#     os.makedirs(test_folder, exist_ok=True)
#
#     # 循环处理每个xlsx文件
#     for i, xlsx_file in enumerate(xlsx_files):
#         # 读取xlsx文件数据
#         data = pd.read_excel(xlsx_file)
#
#         # 去掉第一行的文字
#         # -----> 修改部分：将 [1:] 修改为 [1:].reset_index(drop=True)
#         data = data.iloc[1:].reset_index(drop=True)
#
#         # 提取需要的数据列
#         # -----> 修改部分：将 [100:1300] 修改为 [99:1299]
#         data = data.iloc[99:1299]
#
#         # 根据实验次数和文件名构建新的文件名
#         experiment_num = i + 1
#         new_file_name = f'{bacteria_name}_experiment_{experiment_num}.txt'
#
#         # 写入处理后的数据到新的文件中
#         new_file_path = os.path.join(final_data_folder, new_file_name)
#         data.to_csv(new_file_path, sep='\t', index=False)
#
#         print(f'Successfully processed {xlsx_file} to {new_file_path}')
#
#         # 根据文件序号判断是否将文件分配到训练集或测试集文件夹中
#         if i < num_train_files:
#             shutil.move(new_file_path, train_folder)
#         else:
#             shutil.move(new_file_path, test_folder)
#
# print('Data preprocessing and file allocation completed.')

# import os
# import glob
# import shutil
# import random
# import pandas as pd

# # 设置原始数据文件夹和最终数据文件夹路径
# origin_data_folder = './Origin_Data2'
# final_data_folder = './Final_Data2'

# # 获取所有细菌种类文件夹的路径
# bacteria_folders = glob.glob(os.path.join(origin_data_folder, '*'))

# # 循环处理每个细菌种类文件夹
# for bacteria_folder in bacteria_folders:
#     # 提取细菌名称
#     bacteria_name = os.path.basename(bacteria_folder)

#     # 获取该细菌种类文件夹下所有xlsx文件的路径
#     xlsx_files = glob.glob(os.path.join(bacteria_folder, '*.xlsx'))

#     # 打乱xlsx_files的顺序
#     random.shuffle(xlsx_files)

#     # 计算训练集和测试集的分割点
#     num_files = len(xlsx_files)
#     train_ratio = 0.8
#     num_train_files = int(num_files * train_ratio)

#     # 创建训练集和测试集文件夹
#     train_folder = os.path.join(final_data_folder, 'train')
#     test_folder = os.path.join(final_data_folder, 'test')
#     os.makedirs(train_folder, exist_ok=True)
#     os.makedirs(test_folder, exist_ok=True)

#     # 循环处理每个xlsx文件
#     for i, xlsx_file in enumerate(xlsx_files):
#         # 读取xlsx文件数据
#         data = pd.read_excel(xlsx_file)

#         # 去掉第一行的文字
#         # -----> 修改部分：将 [1:] 修改为 [1:].reset_index(drop=True)
#         data = data.iloc[1:].reset_index(drop=True)

#         # 提取需要的数据列
#         # -----> 修改部分：将 [100:1300] 修改为 [99:1299]
#         data = data.iloc[99:1499]

#         # 根据实验次数和文件名构建新的文件名
#         experiment_num = i + 1
#         new_file_name = f'{bacteria_name}_experiment_{experiment_num}.txt'

#         # 写入处理后的数据到新的文件中
#         new_file_path = os.path.join(final_data_folder, new_file_name)
#         data.to_csv(new_file_path, sep='\t', index=False)

#         print(f'Successfully processed {xlsx_file} to {new_file_path}')

#         # 根据文件序号判断是否将文件分配到训练集或测试集文件夹中
#         if i < num_train_files:
#             shutil.move(new_file_path, train_folder)
#         else:
#             shutil.move(new_file_path, test_folder)

# print('Data preprocessing and file allocation completed.')
import os
import glob
import shutil
import random
import pandas as pd

# 设置原始数据文件夹和最终数据文件夹路径
origin_data_folder = './Origin_Data/data2'
final_data_folder = './Final_Data_Ori'

# 获取所有细菌种类文件夹的路径
bacteria_folders = glob.glob(os.path.join(origin_data_folder, '*'))

# 循环处理每个细菌种类文件夹
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
    num_train_files = int(num_files * train_ratio)

    # 创建训练集和测试集文件夹
    train_folder = os.path.join(final_data_folder, 'train')
    test_folder = os.path.join(final_data_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 循环处理每个txt文件
    for i, txt_file in enumerate(txt_files):
        # 读取txt文件数据
        data = pd.read_csv(txt_file, sep='\t')

        # 去掉第一行的文字
        data = data.iloc[2:]

        # 提取后两列数字
        # data = data.iloc[-3:]

        # 根据实验次数和文件名构建新的文件名
        experiment_num = i + 1
        new_file_name = f'{bacteria_name}_experiment_{experiment_num}.txt'

        # 写入处理后的数据到新的文件中
        new_file_path = os.path.join(final_data_folder, new_file_name)
        data.to_csv(new_file_path, sep='\t', index=False)

        print(f'Successfully processed {txt_file} to {new_file_path}')

        # 根据文件序号判断是否将文件分配到训练集或测试集文件夹中
        if i < num_train_files:
            shutil.move(new_file_path, train_folder)
        else:
            shutil.move(new_file_path, test_folder)

print('Data preprocessing and file allocation completed.')
