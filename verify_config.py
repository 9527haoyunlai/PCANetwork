#!/usr/bin/env python3
"""验证PCAN NTU-60优化配置是否正确"""

import sys
from mmengine.config import Config

def verify_config():
    """验证配置文件的关键优化项"""
    
    config_path = 'configs/skeleton/posec3d/rgbpose_conv3d/pcan_ntu60.py'
    print(f"\n{'='*60}")
    print(f"验证配置文件: {config_path}")
    print(f"{'='*60}\n")
    
    try:
        cfg = Config.fromfile(config_path)
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False
    
    all_passed = True
    
    # 1. 检查max_epochs
    max_epochs = cfg.train_cfg.get('max_epochs', 0)
    if max_epochs == 50:
        print(f"✅ max_epochs = {max_epochs} (优化完成)")
    else:
        print(f"❌ max_epochs = {max_epochs} (应为50)")
        all_passed = False
    
    # 2. 检查学习率
    lr = cfg.optim_wrapper.optimizer.lr
    if lr == 0.01:
        print(f"✅ 初始学习率 = {lr} (优化完成)")
    else:
        print(f"⚠️  初始学习率 = {lr} (建议0.01)")
    
    # 3. 检查梯度裁剪
    max_norm = cfg.optim_wrapper.clip_grad.get('max_norm', 0)
    if max_norm == 20:
        print(f"✅ 梯度裁剪 max_norm = {max_norm} (优化完成)")
    else:
        print(f"⚠️  梯度裁剪 max_norm = {max_norm} (建议20)")
    
    # 4. 检查学习率调度器
    schedulers = cfg.param_scheduler
    if len(schedulers) == 2:
        if schedulers[0]['type'] == 'LinearLR' and schedulers[1]['type'] == 'CosineAnnealingLR':
            print(f"✅ 学习率调度: LinearLR + CosineAnnealingLR (优化完成)")
        else:
            print(f"⚠️  学习率调度类型不匹配")
            all_passed = False
    else:
        print(f"❌ 学习率调度器数量 = {len(schedulers)} (应为2)")
        all_passed = False
    
    # 5. 检查loss权重
    loss_weights = cfg.model.cls_head.get('loss_weights', [])
    if len(loss_weights) == 4:
        if loss_weights[1] == 2.0:  # pose分支权重
            print(f"✅ Loss权重 = {loss_weights} (Pose分支已提升)")
        else:
            print(f"⚠️  Loss权重 = {loss_weights} (Pose分支建议为2.0)")
    else:
        print(f"❌ Loss权重配置异常")
        all_passed = False
    
    # 6. 检查resume配置
    resume = cfg.get('resume', False)
    if resume:
        print(f"✅ resume = True (将从checkpoint继续训练)")
    else:
        print(f"⚠️  resume = False (将从头训练)")
    
    # 7. 检查早停配置
    if hasattr(cfg, 'default_hooks') and 'early_stopping' in cfg.default_hooks:
        es_cfg = cfg.default_hooks.early_stopping
        patience = es_cfg.get('patience', 0)
        monitor = es_cfg.get('monitor', '')
        print(f"✅ 早停机制已配置: patience={patience}, monitor={monitor}")
    else:
        print(f"⚠️  未检测到早停配置")
    
    # 8. 检查checkpoint配置
    if hasattr(cfg, 'default_hooks') and 'checkpoint' in cfg.default_hooks:
        ckpt_cfg = cfg.default_hooks.checkpoint
        interval = ckpt_cfg.get('interval', 0)
        save_best = ckpt_cfg.get('save_best', '')
        print(f"✅ Checkpoint配置: interval={interval}, save_best={save_best}")
    else:
        print(f"⚠️  Checkpoint配置使用默认值")
    
    # 9. 检查数据增强
    train_pipeline = cfg.train_dataloader.dataset.pipeline
    has_colorjitter = any(t.get('type') == 'ColorJitter' for t in train_pipeline)
    if has_colorjitter:
        print(f"✅ 数据增强已添加ColorJitter")
    else:
        print(f"⚠️  未检测到ColorJitter增强")
    
    # 总结
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ 配置验证通过！所有优化项已正确配置")
        print("\n🚀 可以开始训练了！运行命令:")
        print("   ./train_resume.sh")
    else:
        print("⚠️  部分配置需要检查，请参考上述提示")
    print(f"{'='*60}\n")
    
    # 显示关键配置摘要
    print("\n📊 关键配置摘要:")
    print(f"  Epochs: {max_epochs}")
    print(f"  Learning Rate: {lr} (warmup: 5 epochs)")
    print(f"  Scheduler: {schedulers[0]['type']} + {schedulers[1]['type']}")
    print(f"  Loss Weights: {loss_weights}")
    print(f"  Gradient Clip: {max_norm}")
    print(f"  Resume: {resume}")
    print(f"  Batch Size: {cfg.train_dataloader.batch_size}")
    print()
    
    return all_passed

if __name__ == '__main__':
    success = verify_config()
    sys.exit(0 if success else 1)

