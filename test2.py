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

    batch_size = 256
    lr = 0.0002
    lamda = 0.3

    name = "PTTE2"
    subname = f"PTTE2_bs-{batch_size}_lr-{lr}_lamda{lamda}_sd-{seed}"  #

    model_path = f"./models/{name}_{subname}/"
    config = generate_config()
    model = PTTE(*config).to(device)  # 模型

    model.load_state_dict(torch.load(model_path + f"{name}_{subname}_best.pkl"))

    dataset_test = TrajDataset2("./data16_test_cc.csv")
    print(f"dataset_test: {len(dataset_test)}")

    # test MAPE, MAE, MSE, RMSE
    model.eval()
    total_loss_mape = 0.0
    total_loss_mae = 0.0
    total_loss_mse = 0.0
    total_loss_rmse = 0.0
    criterion_mape = MAPE().to(device)
    criterion_mae = MAE().to(device)
    criterion_mse = MSE().to(device)
    t1 = time.time()
    for inputs, link_len, labels in traj_dataloader(dataset_test, batch_size, False):
        inputs = [i.to(device) for i in inputs]  # 放到gpu里
        labels1 = labels[0].to(device)  # 放到gpu里
        labels2 = labels[1].squeeze(1).to(device)  # 放到gpu里
        with torch.no_grad():
            preditions = model(*inputs, link_len)
            loss_mape = criterion_mape(preditions[0], labels1)
            loss_mae = criterion_mae(preditions[0], labels1)
            loss_mse = criterion_mse(preditions[0], labels1)
        total_loss_mape += loss_mape.cpu().item() * len(link_len)
        total_loss_mae += loss_mae.cpu().item() * len(link_len)
        total_loss_mse += loss_mse.cpu().item() * len(link_len)
    avg_loss_mape = total_loss_mape / len(dataset_test)
    avg_loss_mae = total_loss_mae / len(dataset_test)
    avg_loss_mse = total_loss_mse / len(dataset_test)

    print(get_duration(t1))
    print(f"TEST_LOSS MAPE: {avg_loss_mape}")
    print(f"TEST_LOSS MAE : {avg_loss_mae}")
    print(f"TEST_LOSS MSE : {avg_loss_mse}")

