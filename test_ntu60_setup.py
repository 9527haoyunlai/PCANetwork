import torch
import pickle
import numpy as np
import sys
sys.path.insert(0, '/home/zh/ChCode/codes01/mmaction2')

print("="*60)
print("NTU-60完整检查")
print("="*60)

# 1. 检查pkl数据
print("\n1. 检查pkl数据...")
with open('data/skeleton/ntu60_xsub.pkl', 'rb') as f:
    data = pickle.load(f)

labels = [ann['label'] for ann in data['annotations']]
print(f"   标签范围: [{min(labels)}, {max(labels)}]")
print(f"   样本数: {len(labels)}")

if min(labels) != 0 or max(labels) != 59:
    print(f"   ✗ 错误：标签应该是0-59！")
    exit(1)
else:
    print(f"   ✓ 标签范围正确")

# 2. 检查函数导入
print("\n2. 检查函数导入...")
try:
    from mmaction.models.heads.rgbpose_head import action2body_ntu60
    from mmaction.models.recognizers.recognizer3d_mm import fine2coarse_ntu60
    print("   ✓ 所有函数导入成功")
except ImportError as e:
    print(f"   ✗ 导入失败: {e}")
    exit(1)

# 3. 测试映射函数
print("\n3. 测试映射函数...")
test_labels = np.array([0, 10, 20, 30, 40, 50, 59])
mapped = [action2body_ntu60(int(x)) for x in test_labels]
print(f"   Action: {test_labels.tolist()}")
print(f"   Body:   {mapped}")

if max(mapped) <= 7 and min(mapped) >= 0:
    print("   ✓ 映射结果在0-7范围内")
else:
    print(f"   ✗ 映射越界！")
    exit(1)

# 4. 模拟一个batch的处理
print("\n4. 模拟batch处理...")
batch_labels = torch.tensor([0, 15, 30, 45, 59], dtype=torch.long)
print(f"   模拟标签: {batch_labels.tolist()}")

max_label = int(batch_labels.max().item())
print(f"   最大标签: {max_label}")

is_ntu60 = (max_label >= 52)
print(f"   判定为NTU-60: {is_ntu60}")

labels_body = batch_labels.cpu().numpy()
if is_ntu60:
    labels_body = np.array([action2body_ntu60(int(i)) for i in labels_body])
else:
    labels_body = np.array([action2body(int(i)) for i in labels_body])

print(f"   粗分类结果: {labels_body.tolist()}")
print(f"   粗分类范围: [{labels_body.min()}, {labels_body.max()}]")

# 5. 测试one_hot
print("\n5. 测试one_hot编码...")
try:
    labels_tensor = torch.tensor(labels_body, dtype=torch.long).cuda()
    one_hot = torch.nn.functional.one_hot(labels_tensor, num_classes=8)
    print(f"   ✓ one_hot成功: {one_hot.shape}")
except Exception as e:
    print(f"   ✗ one_hot失败: {e}")
    exit(1)

print("\n" + "="*60)
print("✓ 所有检查通过！可以开始训练")
print("="*60)
