import importlib
from typing import Type

from classeg.training.default_trainers.classification_trainer import ClassificationTrainer
from classeg.training.default_trainers.segmentation_trainer import SegmentationTrainer
from classeg.training.default_trainers.self_supervised_trainer import SelfSupervisedTrainer
from classeg.training.trainer import Trainer
import glob
import os.path
import click
import multiprocessing_logging
import shutil
from classeg.utils.constants import *
from classeg.utils.utils import get_dataset_name_from_id, import_from_recursive, get_dataset_mode_from_name
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import datetime


def setup_ddp(rank: int, world_size: int) -> None:
    """
    Prepares the ddp on a specific process.
    :param rank: The device we are initializing.
    :param world_size: The total number of devices.
    :return: None
    """
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "12345"
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


def ddp_training(rank, world_size: int, dataset_id: int,
                 fold: int, model: str,
                 session_id: str, resume: bool,
                 config: str, preload: bool, trainer_class: Type[Trainer], dataset_desc: str) -> None:
    """
    Launches training on a single process using pytorch ddp.
    :param preload:
    :param config: The name of the config to load.
    :param session_id: Session id to be used for folder name on output.
    :param rank: The rank we are starting.
    :param world_size: The total number of devices
    :param dataset_id: The dataset to train on
    :param fold: The fold to train
    :param model: The path to the model json definition
    :param resume: Continue training from latest epoch
    :param trainer_class: Trainer class to use
    :param dataset_desc: Trainer class to useZ
    :return: Nothing
    """
    setup_ddp(rank, world_size)
    dataset_name = get_dataset_name_from_id(dataset_id, dataset_desc)
    trainer = None
    try:
        trainer = trainer_class(
            dataset_name,
            fold,
            model,
            rank,
            session_id,
            config,
            resume=resume,
            preload=preload, world_size=world_size)
        trainer.train()
    except Exception as e:
        if trainer is not None and trainer.output_dir is not None:
            out_files = glob.glob(f"{trainer.output_dir}/*")
            if len(out_files) < 4:
                shutil.rmtree(trainer.output_dir, ignore_errors=True)
        raise e
    destroy_process_group()


@click.command()
@click.option("-fold", "-f", help="Which fold to train.", type=int, required=True)
@click.option("-dataset_id", "-d", help="The dataset id to train.", type=str, required=True)
@click.option("-model", "-m", help="Path to model json definition.", type=str, required=True)
@click.option("--gpus", "-g", help="How many gpus for ddp", type=int, default=1)
@click.option("--resume", "--r", help="Resume training from latest", type=bool, is_flag=True)
@click.option("-config", "-c", help="Name of the config file to utilize.", type=str, default="config")
@click.option("--preload", "--p", help="Should the datasets preload.", is_flag=True, type=bool)
@click.option("-name", "-n", help="Output folder name.", type=str, default=None)
@click.option("-extension", "-ext", help="Name of the extension that you want to use.", type=str, default=None)
@click.option("-dataset_desc", "-dd", required=False,
              help="Description of dataset. Useful if you have overlapping ids.")  # 10
def main(
        fold: int,
        dataset_id: str,
        model: str,
        gpus: int,
        resume: bool,
        config: str,
        preload: bool,
        name: str,
        extension: str,
        dataset_desc: str
) -> None:
    """
    Initializes training on multiple processes, and initializes logger.
    :param preload: Should datasets preload
    :param config: The name oof the config file to load.
    :param gpus: How many gpus to train with
    :param dataset_id: The dataset to train on
    :param fold: The fold to train
    :param model: The path to the model json definition
    :param resume: The weights to load, or None
    :param name: The name of the output folder. Will timestamp by default.
    :param extension: The name of the trainer class to use
    :param dataset_desc: Dataset description
    :return:
    """
    multiprocessing_logging.install_mp_handler()
    if 'json' not in model:
        # try to find it in the default model bucket
        available_models = glob.glob(f"{MODEL_BUCKET_DIRECTORY}/**/*", recursive=True)
        for model_path in available_models:
            if model_path.split('/')[-1].split('.')[0] == model:
                model = model_path
                break

    if not os.path.exists(model):
        raise ValueError("The model you specified doesn't exist. We checked if it was a full path, and if it is in "
                         "the default model bucket."
                         "It is indeed not.")

    mode = get_dataset_mode_from_name(get_dataset_name_from_id(dataset_id, dataset_desc))
    if extension is not None:
        module = importlib.import_module(f"classeg.extensions.{extension}")
        trainer_name = getattr(module, "TRAINER_CLASS_NAME")
        trainer_class = import_from_recursive(f"classeg.extensions.{extension}.training", trainer_name)
    else:
        trainer_class = {
            SEGMENTATION: SegmentationTrainer,
            SELF_SUPERVISED: SelfSupervisedTrainer,
            CLASSIFICATION: ClassificationTrainer
        }[mode]
    print(f"Training detected mode {mode}")
    # This sets the behavior of some modules in json models utils.
    session_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%f") if name is None else name
    if gpus > 1:
        mp.spawn(
            ddp_training,
            args=(
                gpus,
                dataset_id,
                fold,
                model,
                session_id,
                resume,
                config,
                preload,
                trainer_class,
                dataset_desc
            ),
            nprocs=gpus,
            join=True,
        )
    elif gpus == 1:
        dataset_name = get_dataset_name_from_id(dataset_id, dataset_desc)
        trainer = None
        try:
            trainer = trainer_class(
                dataset_name,
                fold,
                model,
                0,
                session_id,
                config,
                resume=resume,
                preload=preload,
                world_size=1
            )
            trainer.train()
        except Exception as e:
            if trainer is not None and trainer.output_dir is not None:
                out_files = glob.glob(f"{trainer.output_dir}/*")
                if len(out_files) < 4:
                    shutil.rmtree(trainer.output_dir, ignore_errors=True)
            raise e
    else:
        raise NotImplementedError("You must set gpus to >= 1")


if __name__ == "__main__":
    main()