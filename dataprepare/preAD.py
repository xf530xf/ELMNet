import os
import shutil

# 设置主目录路径和目标文件夹路径
main_folder = "/data/fengxiao/UltraLight-VM-UNet/data/Fish_Dataset"  # 替换为你的主文件夹路径


# 输出文件夹
output_image_folder = os.path.join(main_folder, "image")
output_gt_folder = os.path.join(main_folder, "GT")

# 创建输出文件夹（如果不存在）
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_gt_folder, exist_ok=True)

# 计数器
image_counter = 1
gt_counter = 1

# 支持的图片格式
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def copy_images_recursively(source_root, output_root, counter_start):
    counter = counter_start
    for root, dirs, files in os.walk(source_root):
        for file in sorted(files):
            if file.lower().endswith(valid_exts):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(output_root, f"{counter}.png")
                shutil.copy(src_file, dst_file)
                counter += 1
    return counter

# 遍历主文件夹下的每个子文件夹
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    test_path = os.path.join(subfolder_path, "image")
    gt_path = os.path.join(subfolder_path, "GT")

    if os.path.exists(test_path):
        image_counter = copy_images_recursively(test_path, output_image_folder, image_counter)

    if os.path.exists(gt_path):
        gt_counter = copy_images_recursively(gt_path, output_gt_folder, gt_counter)

print("所有图片已成功复制并重新编号！")

# print("图片整理完成！")