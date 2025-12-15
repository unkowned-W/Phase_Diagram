import pandas as pd
import os
import glob

def process_single_file(csv_file_path):
    """处理单个CSV文件"""
    # 读取文件
    df = pd.read_csv(csv_file_path)
    df['source_file'] = os.path.basename(csv_file_path)
    
    # 确保必要的列存在
    required_columns = ['Gd104', 'Gd42', 'source_file']
    for col in ['Gd104', 'Gd42']:
        if col not in df.columns:
            raise ValueError(f"文件 {csv_file_path} 中缺少列: {col}")
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # 确定特征列
    feature_columns = [col for col in df.columns if col not in required_columns]
    
    # 第一阶段：是否有晶体
    df_stage1 = df[feature_columns].copy()
    df_stage1['Y'] = ((df['Gd104'] == 1) | (df['Gd42'] == 1)).astype(int)
    df_stage1['source_file'] = df['source_file'] 

    # 第二阶段：晶体类型（仅包含有晶体的样本）
    has_crystal = (df['Gd104'] == 1) | (df['Gd42'] == 1)
    df_stage2 = df[has_crystal][feature_columns].copy()
    df_stage2['Y'] = 0  # 默认为Gd104
    df_stage2.loc[df[has_crystal]['Gd42'] == 1, 'Y'] = 1
    df_stage2['source_file'] = df[has_crystal]['source_file'].values  # 保留源文件信息

    return df_stage1, df_stage2, df

def process_folder(folder_path, recursive=False):
    """处理整个文件夹中的所有CSV文件，返回合并后的数据"""
    # 获取所有CSV文件
    if recursive:
        csv_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    else:
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"警告：在 {folder_path} 中未找到CSV文件")
        return pd.DataFrame(), pd.DataFrame()
    
    # 存储每个文件的结果
    stage1_list = []
    stage2_list = []
    combined_list = []
    # 处理每个文件
    for csv_file in csv_files:
        try:
            file_name = os.path.basename(csv_file)
            print(f"处理文件: {file_name}")
            
            df_stage1, df_stage2, combined_df = process_single_file(csv_file)
            
            stage1_list.append(df_stage1)
            stage2_list.append(df_stage2)
            combined_list.append(combined_df)

            # 显示统计信息
            print(f" 样本数: {len(combined_df)}，有晶体: {df_stage1['Y'].sum()}，Gd42: {(df_stage2['Y'] == 1).sum()}")
            
        except Exception as e:
            print(f"处理文件 {csv_file} 失败: {e}")
    
    # 合并所有文件的结果
    if stage1_list:
        combined_stage1_df = pd.concat(stage1_list, ignore_index=True)
        combined_stage2_df = pd.concat(stage2_list, ignore_index=True)
        combined_all_df = pd.concat(combined_list, ignore_index=True)
        print("\n" + "="*50)
        print("文件夹处理完成！")
        print(f"共处理 {len(stage1_list)} 个文件")
        print(f"合并后总样本数: {len(combined_all_df)}")
        print(f"总有晶体样本: {combined_stage1_df['Y'].sum()} ({combined_stage1_df['Y'].sum()/len(combined_stage1_df)*100:.1f}%)")
        
        if len(combined_stage2_df) > 0:
            gd42_count = (combined_stage2_df['Y'] == 1).sum()
            print(f"总Gd42晶体: {gd42_count} ({gd42_count/len(combined_stage2_df)*100:.1f}%)")
        print("="*50)
        
        return combined_stage1_df, combined_stage2_df
    