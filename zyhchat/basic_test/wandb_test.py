import wandb

# 初始化一个实验
wandb.init(project="my-first-wandb", name="exp-01")

# 模拟训练过程
for epoch in range(10):
    loss = 1.0 / (epoch + 1)  # 模拟 loss 下降
    acc = 0.8 + epoch * 0.02  # 模拟 accuracy 上升
    
    # 记录指标
    wandb.log({
        "loss": loss,
        "accuracy": acc,
        "epoch": epoch
    })

wandb.finish()  # 结束记录