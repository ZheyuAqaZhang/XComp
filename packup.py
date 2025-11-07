import os
import shutil

'''
新建 lmms-eval_videochat_packup 的文件夹，把 lmms_eval_videochat 里面的所有 py 文件都放进去，包括多级路径下的
'''

src_dir = "lmms-eval_videochat"
dst_dir = "lmms-eval_videochat_packup"

os.makedirs(dst_dir, exist_ok=True)

for root, dirs, files in os.walk(src_dir):
    # print(f"root: {root} dirs: {dirs} files: {files}")
    for f in files:
        if f.endswith(".py"):
            rel_path = os.path.relpath(os.path.join(root, f), src_dir)
            dst_file = os.path.join(dst_dir, rel_path)
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy(os.path.join(root, f), dst_file)