import pickle
from gears import PertData
from gears.inference import *
from util import parse_args, set_seed

class Predictor:
    def __init__(self, pert_data, model_config={}):
        self.train_data = pert_data.adata[pert_data.adata.obs["condition"].isin(pert_data.set2conditions["train"])]
        self.valid_data = pert_data.adata[pert_data.adata.obs["condition"].isin(pert_data.set2conditions["val"])]
        ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        self.basis = np.mean(ctrl_adata.X.toarray(), axis=0)
        self.estimator = []
        for cond in pert_data.set2conditions["train"]:
            if cond != "ctrl":
                cond_adata = pert_data.adata[pert_data.adata.obs["condition"] == cond]
                cond_data = np.mean(cond_adata.X.toarray(), axis=0)
                self.estimator.append(cond_data - self.basis)
        self.estimator = np.mean(np.array(self.estimator), axis=0)

def gears_evaluate(model, loader):
    results, pert_cat = {}, []
    pred, truth = [], []
    pred_de, truth_de = [], []

    with torch.no_grad():
        for itr, batch in enumerate(loader):
            t = batch.y
            p = batch.x.view(t.shape) + model.estimator
            pert_cat.extend(batch.pert)
            pred.extend(p.cpu())
            truth.extend(t.cpu())
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"]= pred.detach().cpu().numpy()
    results["truth"]= truth.detach().cpu().numpy()
    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"]= pred_de.detach().cpu().numpy()
    results["truth_de"]= truth_de.detach().cpu().numpy()
    return results

def gears_test(model, pert_data):
    test_loader = pert_data.dataloader["test_loader"]
    test_res = gears_evaluate(model, test_loader)
    test_metrics, test_pert_res = compute_metrics(test_res)
    log = "Best performing model: Test Top 20 DE MSE: {:.4f}"
    print(log.format(test_metrics["mse_de"]))

    out = deeper_analysis(pert_data.adata, test_res)
    out_non_dropout = non_dropout_analysis(pert_data.adata, test_res)
    metrics = ["pearson_delta"]
    metrics_non_dropout = ["frac_opposite_direction_top20_non_dropout",
                            "frac_sigma_below_1_non_dropout",
                            "mse_top20_de_non_dropout"]

    subgroup = pert_data.subgroup
    subgroup_analysis = {}
    for name in subgroup["test_subgroup"].keys():
        subgroup_analysis[name] = {}
        for m in list(list(test_pert_res.values())[0].keys()):
            subgroup_analysis[name][m] = []

    for name, pert_list in subgroup["test_subgroup"].items():
        for pert in pert_list:
            for m, res in test_pert_res[pert].items():
                subgroup_analysis[name][m].append(res)

    for name, result in subgroup_analysis.items():
        for m in result.keys():
            subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
            print("test_" + name + "_" + m + ": " + str(subgroup_analysis[name][m]))

    subgroup_analysis = {}
    for name in subgroup["test_subgroup"].keys():
        subgroup_analysis[name] = {}
        for m in metrics:
            subgroup_analysis[name][m] = []
        for m in metrics_non_dropout:
            subgroup_analysis[name][m] = []

    for name, pert_list in subgroup["test_subgroup"].items():
        for pert in pert_list:
            for m in metrics:
                subgroup_analysis[name][m].append(out[pert][m])
            for m in metrics_non_dropout:
                subgroup_analysis[name][m].append(out_non_dropout[pert][m])

    for name, result in subgroup_analysis.items():
        for m in result.keys():
            subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
            print("test_" + name + "_" + m + ": " + str(subgroup_analysis[name][m]))

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    pert_data = PertData(args.data_path)
    pert_data.load(data_name=args.data_name)
    pert_data.prepare_split(split="simulation", seed=args.seed)
    pert_data.get_dataloader(batch_size=32)
    model = Predictor(pert_data=pert_data)
    gears_test(model, pert_data)
