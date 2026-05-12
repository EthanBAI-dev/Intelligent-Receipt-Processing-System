import os
from PIL import Image
from pillow_heif import register_heif_opener

# 1. 注册 HEIF 解码器
register_heif_opener()


def batch_convert_with_diagnostics(input_folder, output_folder):
    # --- 诊断步骤：打印绝对路径 ---
    abs_input = os.path.abspath(input_folder)
    abs_output = os.path.abspath(output_folder)

    print(f"🔍 正在尝试读取文件夹: {abs_input}")
    print(f"🔍 正在尝试保存至文件夹: {abs_output}")
    print("-" * 30)

    # 检查输入文件夹是否存在
    if not os.path.exists(abs_input):
        print(f"❌ 错误：找不到输入文件夹！请检查路径是否正确。")
        return

    if not os.path.exists(abs_output):
        os.makedirs(abs_output)
        print(f"✅ 已创建输出目录: {abs_output}")

    count = 0
    # 获取文件列表并过滤 HEIC
    files = [f for f in os.listdir(abs_input) if f.lower().endswith(".heic")]

    if not files:
        print(f"⚠️ 警告：在输入文件夹中没有找到任何 .heic 文件。")
        return

    for filename in files:
        heic_path = os.path.join(abs_input, filename)
        try:
            img = Image.open(heic_path)
            img = img.convert("RGB")  # 必须转 RGB 才能存为 JPG

            target_filename = os.path.splitext(filename)[0] + ".jpg"
            target_path = os.path.join(abs_output, target_filename)

            img.save(target_path, "JPEG", quality=95)
            count += 1
            print(f"成功 [{count}]: {filename} -> {target_filename}")
        except Exception as e:
            print(f"❌ 转换 {filename} 失败: {e}")

    print(f"\n✨ 任务完成！总共转换了 {count} 张图片。")
    print(f"📂 最终输出位置: {abs_output}")


# =========================
# 路径设置（使用 ~ 符号最稳妥）
# =========================
# os.path.expanduser 会自动把 ~ 替换成 /Users/baiwenbin
raw_input = os.path.expanduser("~/Documents/MyProjects/DeepLearning Project/work")
raw_output = os.path.expanduser("~/Documents/MyProjects/DeepLearning Project/work/processed data2")

batch_convert_with_diagnostics(raw_input, raw_output)