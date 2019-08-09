from second.pytorch.train import train, evaluate
from google.protobuf import text_format
from second.protos import pipeline_pb2
from pathlib import Path
from second.utils import config_tool, model_tool
import datetime
from second.data.all_dataset import get_dataset_class

# for yj.star local path
PRETRAINED_MODELS_FOLDER = "/data/second_v1.6_trained_model/kitti/pretrained_models_v1.5/pp_model_for_nuscenes_pretrain"
KITTI_DATASET_ROOT = "/data/KITTI_DATASET_ROOT"
NUSCENES_DATASET_ROOT = "/data/NUSCENES_DATASET_ROOT"


def _div_up(a, b):
    return (a + b - 1) // b


def _get_config(path):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    return config
                

def _nuscenes_modify_step(config,
                          epochs,
                          eval_epoch,
                          data_sample_factor,
                          num_examples=28130):
    input_cfg = config.train_input_reader
    train_cfg = config.train_config
    batch_size = input_cfg.batch_size
    data_sample_factor_to_name = {
        1: "NuScenesDataset",
        2: "NuScenesDatasetD2",
        3: "NuScenesDatasetD3",
        4: "NuScenesDatasetD4",
        5: "NuScenesDatasetD5",
        6: "NuScenesDatasetD6",
        7: "NuScenesDatasetD7",
        8: "NuScenesDatasetD8",
    }
    dataset_name = data_sample_factor_to_name[data_sample_factor]
    input_cfg.dataset.dataset_class_name = dataset_name
    ds = get_dataset_class(dataset_name)(
        root_path=input_cfg.dataset.kitti_root_path,
        info_path=input_cfg.dataset.kitti_info_path,
    )
    num_examples_after_sample = len(ds)
    step_per_epoch = _div_up(num_examples_after_sample, batch_size)
    step_per_eval = step_per_epoch * eval_epoch
    total_step = step_per_epoch * epochs
    train_cfg.steps = total_step
    train_cfg.steps_per_eval = step_per_eval


def train_nuscenes_lite():
    config = Path(
        __file__).resolve().parent / "configs/nuscenes/car.lite.nu.config"
    ckpt_path = "/home/yy/deeplearning/voxelnet_torch_sparse/car_lite_small_v1/voxelnet-15500.tckpt"
    # config = Path(__file__).resolve().parent() / "configs/car.fhd.nu.config"
    config = _get_config(config)
    _nuscenes_modify_step(config, 50, 5, 8)
    model_dir_root = Path("/home/yy/deeplearning/model_dirs/nuscene")
    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    train(
        config,
        model_dir_root / "car_lite_with_pretrain" / ("test_" + date_str),
        pretrained_path=ckpt_path)


def train_nuscenes():
    config = Path(
        __file__).resolve().parent / "configs/nuscenes/car.fhd.nu.config"
    ckpt_path = "/home/yy/deeplearning/voxelnet_torch_sparse/car_fhd_small_v1/voxelnet-27855.tckpt"
    # config = Path(__file__).resolve().parent() / "configs/car.fhd.nu.config"
    config = _get_config(config)
    _nuscenes_modify_step(config, 50, 5, 8)
    model_dir_root = Path("/home/yy/deeplearning/model_dirs/nuscene")
    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    train(
        config,
        model_dir_root / "car_fhd_with_pretrain" / ("test_" + date_str),
        pretrained_path=ckpt_path)


def train_nuscenes_pp():
    config = Path(
        __file__).resolve().parent / "configs/nuscenes/pp.nu.config"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    # config = Path(__file__).resolve().parent() / "configs/car.fhd.nu.config"
    config = _get_config(config)
    _nuscenes_modify_step(config, 50, 5, 8)
    model_dir_root = Path("/home/yy/deeplearning/model_dirs/nuscene")
    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    train(
        config,
        model_dir_root / "car_pp_with_pretrain" / ("test_" + date_str),
        pretrained_path=ckpt_path)

def train_nuscenes_all():
    config = Path(
        __file__).resolve().parent / "configs/nuscenes/all.fhd.config"
    ckpt_path = "/home/yy/deeplearning/voxelnet_torch_sparse/car_fhd_small_v1/voxelnet-27855.tckpt"
    # config = Path(__file__).resolve().parent() / "configs/car.fhd.nu.config"
    config = _get_config(config)
    _nuscenes_modify_step(config, 50, 5, 8)
    model_dir_root = Path("/home/yy/deeplearning/model_dirs/nuscene")
    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    train(
        config,
        model_dir_root / "all_fhd" / ("test_" + date_str))

def train_nuscenes_pp_all():
    config = Path(
        __file__).resolve().parent / "configs/nuscenes/all.pp.config"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    # config = Path(__file__).resolve().parent() / "configs/car.fhd.nu.config"
    config = _get_config(config)
    _nuscenes_modify_step(config, 50, 5, 8)
    model_dir_root = Path("/home/yy/deeplearning/model_dirs/nuscene")
    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    train(
        config,
        model_dir_root / "all_pp" / ("test_" + date_str),
        pretrained_path=ckpt_path)

def train_nuscenes_pp_all_sample():
    config = Path(
        __file__).resolve().parent / "configs/nuscenes/all.pp.sample.config"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    # config = Path(__file__).resolve().parent() / "configs/car.fhd.nu.config"
    config = _get_config(config)
    _nuscenes_modify_step(config, 50, 5, 8)
    model_dir_root = Path("/home/yy/deeplearning/model_dirs/nuscene")
    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    train(
        config,
        model_dir_root / "all_pp_sample" / ("test_" + date_str),
        pretrained_path=ckpt_path)

def train_nuscenes_pp_all_sample_v2():
    config = Path(
        __file__).resolve().parent / "configs/nuscenes/all.pp.sample.v2.config"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    # config = Path(__file__).resolve().parent() / "configs/car.fhd.nu.config"
    config = _get_config(config)
    _nuscenes_modify_step(config, 50, 5, 8)
    model_dir_root = Path("/home/yy/deeplearning/model_dirs/nuscene")
    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    train(
        config,
        model_dir_root / "all_pp_sample_v2" / ("test_" + date_str))

def train_nuscenes_pp_all_v2():
    config = Path(
        __file__).resolve().parent / "configs/nuscenes/all.pp.v2.config"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    # config = Path(__file__).resolve().parent() / "configs/car.fhd.nu.config"
    config = _get_config(config)
    _nuscenes_modify_step(config, 50, 5, 8)
    model_dir_root = Path("/home/yy/deeplearning/model_dirs/nuscene")
    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    train(
        config,
        model_dir_root / "all_pp_v2" / ("test_" + date_str))

def train_nuscenes_pp_vel():
    config = Path(
        __file__).resolve().parent / "configs/nuscenes/all.pp.vel.config"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    # config = Path(__file__).resolve().parent() / "configs/car.fhd.nu.config"
    config = _get_config(config)
    _nuscenes_modify_step(config, 50, 5, 8)
    model_dir_root = Path("/home/yy/deeplearning/model_dirs/nuscene")
    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    train(
        config,
        model_dir_root / "pp_vel" / ("test_" + date_str))

def train_nuscenes_pp_vel_v2():
    config = Path(
        __file__).resolve().parent / "configs/nuscenes/all.pp.vel.v2.config"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    # config = Path(__file__).resolve().parent() / "configs/car.fhd.nu.config"
    config = _get_config(config)
    _nuscenes_modify_step(config, 50, 5, 8)
    model_dir_root = Path("/home/yy/deeplearning/model_dirs/nuscene")
    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    train(
        config,
        model_dir_root / "pp_vel" / ("test_" + date_str))

def train_nuscenes_pp_car():
    config = Path(
        __file__).resolve().parent / "configs/nuscenes/car.pp.config"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    ckpt_path = "/home/yy/deeplearning/model_dirs/kitti/car_pp_long_v0/voxelnet-296960.tckpt"
    # config = Path(__file__).resolve().parent() / "configs/car.fhd.nu.config"
    config = _get_config(config)
    _nuscenes_modify_step(config, 50, 5, 8)
    model_dir_root = Path("/home/yy/deeplearning/model_dirs/nuscene")
    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    train(
        config,
        model_dir_root / "pp_car" / ("test_" + date_str))

def train_nuscenes_pp_all_lowa(date_str=None, data_version="v1.0_trainval"):
    config = Path(
        __file__).resolve().parent / ("configs/nuscenes/all.pp.lowa.config")
    ckpt_path = "/data/project/second_v1.6/trained_model/kitti/pretrained_models_v1.5/pp_model_for_nuscenes_pretrain/voxelnet-296960.tckpt"   # need to spread, yj.star
    ckpt_path = None
    # config = Path(__file__).resolve().parent() / "configs/car.fhd.nu.config"
    config = _get_config(config)

    pkl_local_path = ("/data/NUSCENES_DATASET_ROOT/" + data_version + "/pkl_noSweepTimeGap_noMapPrior_v1/") # yj.star, set_here #######################
    config.train_input_reader.dataset.kitti_info_path = (pkl_local_path + "infos_train.pkl")
    config.train_input_reader.preprocess.database_sampler.database_info_path = (pkl_local_path + "kitti_dbinfos_train.pkl")
    config.eval_input_reader.dataset.kitti_info_path = (pkl_local_path + "infos_val.pkl")


    _nuscenes_modify_step(config, 5000, 5, 8)

    model_dir_root = Path("/data/project/second_v1.6/trained_model/nusc")

    if date_str is None:
        date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        is_resume=False
    else:
        is_resume=True

    train(
        config,
        model_dir_root / ("all_pp_lowa_" + data_version + "_noSweepTimeGap_noMapPrior") / ("test_" + date_str),     # yj.star, sethere #######################
        pretrained_path=ckpt_path, multi_gpu=True, resume=is_resume)


if __name__ == "__main__":
    # model_tool.rm_invalid_model_dir("/home/yy/deeplearning/model_dirs/nuscene")
    # train_nuscenes_lite_hrz()
    train_nuscenes_pp_all_lowa(data_version="v1.0-trainval")

    #resume_nuscenes_pp_all()
