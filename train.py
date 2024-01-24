from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

from model import *
from loss_func import *
from dataset import *
from config import *


def get_duration(t1):
    t2 = time.time()
    dt = round(t2 - t1)
    return f"({dt // 3600:02d}:{dt % 3600 // 60:02d}:{dt % 60:02d})"


if __name__ == '__main__':

    seed = 666
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_epochs = 100
    batch_size = 256
    lr = 0.0002

    name = "WDR"
    subname = f"WDR_bs-{batch_size}_lr-{lr}_sd-{seed}"  #

    # 使用"tensorboard --logdir=logs_PTTE --port=6006"查看tensorboard
    use_writer = True  #
    save_model = True
    if use_writer:
        lwriter = open(f"./logs/{name}_{subname}.log", mode='x')
        # swriter = SummaryWriter("logs_" + name)

    model_path = f"./models/{name}_{subname}/"
    config = generate_config()  #
    model = WDR(*config).to(device)  # 模型  #

    dataset_train = TrajDataset("./data16_train.csv")  #
    dataset_val = TrajDataset("./data16_val.csv")
    print(f"dataset_train: {len(dataset_train)}")
    print(f"dataset_val: {len(dataset_val)}")

    criterion_train = MAPE().to(device)  # 损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器

    min_loss = None
    step_count = 0  # 更新了多少步
    step_gap = 100
    t1 = time.time()
    for epoch in tqdm(range(0, n_epochs)):
        # train
        model.train()
        for inputs, link_len, labels in traj_dataloader(dataset_train, batch_size, True):
            inputs = [i.to(device) for i in inputs]  # 放到gpu里
            labels = labels.to(device)  # 放到gpu里
            optimizer.zero_grad()  # 清空梯度
            preditions = model(*inputs, link_len)  # 前向传播
            loss = criterion_train(preditions, labels)  # 计算损失
            loss.backward()  # 后向传播
            optimizer.step()  # 更新参数

            # if step_count % step_gap == 0:
            #     sdt = get_duration(t1)
            #     print(f"STEP_{step_count} TRAIN_LOSS MAPE: {loss.cpu().item()} " + sdt)
            #     if use_writer:
            #         lwriter.write(f"STEP_{step_count} TRAIN_LOSS MAPE: {loss.cpu().item()}" + sdt + "\n")
            #         swriter.add_scalar(f"{name}-{subname}: train_loss_mape",
            #                            loss.cpu().item(), step_count // 100)
            step_count += 1

        # val MAPE
        model.eval()
        total_loss_mape = 0.0
        criterion_mape = MAPE().to(device)
        for inputs, link_len, labels in traj_dataloader(dataset_val, batch_size, False):
            inputs = [i.to(device) for i in inputs]  # 放到gpu里
            labels = labels.to(device)  # 放到gpu里
            with torch.no_grad():
                preditions = model(*inputs, link_len)
                loss_mape = criterion_mape(preditions, labels)
            total_loss_mape += loss_mape.cpu().item() * len(link_len)
        avg_loss_mape = total_loss_mape / len(dataset_val)

        sdt = get_duration(t1)
        print(f"EPOCH_{epoch} VAL_LOSS MAPE: {avg_loss_mape}" + sdt)
        if use_writer:
            lwriter.write(f"{epoch},{avg_loss_mape}\n")
            # lwriter.write(f"EPOCH_{epoch} VAL_LOSS MAPE: {avg_loss_mape} " + sdt + "\n")
            # swriter.add_scalar(f"{name}-{subname}: val_loss_mape", avg_loss_mape, epoch)

        # save model
        if save_model:
            if min_loss is None:
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                min_loss = avg_loss_mape
                path = model_path + f"{name}_{subname}_best.pkl"
                torch.save(model.state_dict(), path)
            else:
                if min_loss > avg_loss_mape:
                    min_loss = avg_loss_mape
                    path = model_path + f"{name}_{subname}_best.pkl"
                    torch.save(model.state_dict(), path)
            # path = model_path + f"{name}_{subname}_ep-{epoch}.pkl"
            # torch.save(model.state_dict(), path)

    if use_writer:
        # swriter.close()  # 关上tensorboard
        lwriter.close()
