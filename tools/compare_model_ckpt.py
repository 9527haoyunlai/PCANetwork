# compare_model_ckpt.py
import torch
from mmengine.config import Config
from mmaction.registry import MODELS

cfg = Config.fromfile('configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d.py')
model = MODELS.build(cfg.model)

# 模型的keys
model_keys = set(model.state_dict().keys())
print(f"模型中的参数数量: {len(model_keys)}")

# Checkpoint的keys
ckpt = torch.load('pretrained/PCAN_checkpoint_fixed.pth', map_location='cpu')
ckpt_keys = set(ckpt['state_dict'].keys())
print(f"Checkpoint中的参数数量: {len(ckpt_keys)}")

# 找出不匹配的
only_in_model = model_keys - ckpt_keys
only_in_ckpt = ckpt_keys - model_keys

print(f"\n只在模型中存在（missing in checkpoint）: {len(only_in_model)}")
if only_in_model:
    print("前10个:")
    for key in list(only_in_model)[:10]:
        print(f"  - {key}")

print(f"\n只在checkpoint中存在（unexpected）: {len(only_in_ckpt)}")
if only_in_ckpt:
    print("前10个:")
    for key in list(only_in_ckpt)[:10]:
        print(f"  - {key}")
        
# 检查PCAN特有模块
pcan_keys = [k for k in ckpt_keys if 'featuredifference' in k or 'fr_' in k]
print(f"\nCheckpoint中的PCAN模块数量: {len(pcan_keys)}")

model_has_pcan = any('featuredifference' in k for k in model_keys)
print(f"模型中是否有PCAN模块: {model_has_pcan}")

if not model_has_pcan and len(pcan_keys) > 0:
    print("\n❌ 严重问题：checkpoint包含PCAN模块，但配置文件构建的模型没有！")
    print("   需要使用PCAN专用的配置文件！")