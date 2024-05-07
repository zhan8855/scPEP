from gears import PertData, GEARS
from util import parse_args, set_seed

args = parse_args()
set_seed(args.seed)

pert_data = PertData(args.data_path)
pert_data.load(data_name=args.data_name)
pert_data.prepare_split(split="simulation", seed=args.seed)
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)

gears_model = GEARS(pert_data, device = "cuda:0")
gears_model.model_initialize(hidden_size = 64)
gears_model.train(epochs = 20)
gears_model.save_model("/path/to/model/dir")
