import os
import argparse

def convert_checkpoints(checkpoint_dir, pretrain_name, tokenizer_name):
    # 定义检查点文件夹路径和其他参数
    last_checkpoint_file = os.path.join(checkpoint_dir, "last_checkpoint")

    # 创建保存转换文件的文件夹
    converted_dir = os.path.join(checkpoint_dir, "converted")
    os.makedirs(converted_dir, exist_ok=True)

    max_step = -1
    latest_ckpt_file = None

    # 遍历文件夹中所有的检查点文件
    for file_name in os.listdir(checkpoint_dir):
        if file_name.endswith(".pt"):
        # 修改 last_checkpoint 文件内容
            with open(last_checkpoint_file, 'w') as file:
                file.write(file_name)
            # 提取 step 数目
            step_number = int(file_name.split('_')[3])

            # 更新最新的ckpt文件
            if step_number > max_step:
                max_step = step_number
                latest_ckpt_file = file_name

            save_name_hf = os.path.join(converted_dir, f"ckpt{step_number}")

            # 如果保存文件已存在，则跳过
            if os.path.exists(save_name_hf):
                print(f"{save_name_hf} already exists. Skipping...")
                continue

            # 构建并执行转换命令
            command = f"""
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python convert_checkpoint_to_hf.py \\
                --tp_ckpt_name {checkpoint_dir} \\
                --pretrain_name {pretrain_name} \\
                --tokenizer_name {tokenizer_name} \\
                --save_name_hf {save_name_hf}
            """
            print(f"Executing command: {command}")
            os.system(command)

    # 在所有转换完成后，修改 last_checkpoint 文件内容
    if latest_ckpt_file:
        with open(last_checkpoint_file, 'w') as file:
            file.write(latest_ckpt_file)
        print(f"Updated {last_checkpoint_file} with {latest_ckpt_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert checkpoints to Hugging Face format')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to the checkpoint directory')
    parser.add_argument('--pretrain_name', type=str, required=True, help='Name of the pre-trained model')
    parser.add_argument('--tokenizer_name', type=str, required=True, help='Name of the tokenizer')

    args = parser.parse_args()

    convert_checkpoints(args.checkpoint_dir, args.pretrain_name, args.tokenizer_name)
