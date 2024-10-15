from yacs.config import CfgNode
from src.data import load_dataset, build_data_loader
from src.model import build_model
from src.model.solver import build_optimizer, build_scheduler, build_evaluator
import time, shutil, os, pickle
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch

TERMINAL = shutil.get_terminal_size().columns


def train(cfg: CfgNode):
    # Prepare Results Folder
    if not cfg.DEBUG:
        assert cfg.OUTPUT_PATH != '', cfg.EXPERIMENT_NAME != ''
        experiment_name = cfg.EXPERIMENT_NAME
        output_path = cfg.OUTPUT_PATH
        name_timestamp = datetime.now().isoformat(sep="T", timespec="auto").replace(":", "_")
        if cfg.DATASET.NAME == "omniglot" and cfg.DATASET.ALPHABET != '':
            experiment_name = cfg.DATASET.ALPHABET
            output_path += '/omniglot_single_task'
        results_path = Path(output_path)/(experiment_name+'_'+name_timestamp)
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        # backup_dir = os.path.join(results_path, 'model_backups')
        # if not os.path.exists(backup_dir):
        #     os.mkdir(backup_dir)
        cfg.dump(stream=open(os.path.join(results_path, f'config.yaml'), 'w'))
        state_fpath = os.path.join(results_path, 'model.pt')
        print(f"\033[93m All Logs & Results Will Be Stored In {results_path} \033[0m".center(TERMINAL))

    # Load Dataset
    if cfg.DATASET.NAME != '':
        train_data, val_data = load_dataset(cfg.DATASET)
    else:
        print("====== Error: Provide Dataset and Its Loader in Data/Datasets Folder ======")
        exit()

    # Build Dataloaders
    train_loader = build_data_loader(train_data, cfg.DATASET, is_training=True, debug=cfg.DEBUG)
    val_loader = build_data_loader(val_data, cfg.DATASET, is_training=False, debug=cfg.DEBUG)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Build Model
    model = build_model(cfg.MODEL)
    model.to(device)

    #Load Configs
    model_cfg = cfg.MODEL
    solver_cfg = model_cfg.SOLVER
    opti_cfg = solver_cfg.OPTIMIZER
    scheduler_cfg = solver_cfg.SCHEDULER

    # Build Optimizer
    optimizer = build_optimizer(model, opti_cfg)

    # Build Scheduler
    scheduler_type = scheduler_cfg.NAME
    scheduler = build_scheduler(optimizer, scheduler_cfg)

    # Build Evaluator
    evaluator = build_evaluator(model_cfg)
    evaluator.float()

    # Load Train Params
    total_epochs = solver_cfg.TOTAL_EPOCHS
    loss_fn = solver_cfg.LOSS.NAME
    current_epoch = 1

    print(" \033[96m ============ Start Training ============ \033[0m".center(TERMINAL))
    s_time = time.time()
    parameters = list(model.parameters())
    for epoch in range(current_epoch, total_epochs+1):
        print(f"\nEpoch: {epoch} ")
        model.train()
        train_itr = tqdm(train_loader)

        for idx, (inputs, labels) in enumerate(train_itr):

            input_data = inputs.float().to(device)
            labels = labels.to(device)

            logits = model(input_data)
            evaluator.set_num_heads(len(logits))

            # Calling MultiHeadsEval forward function to produce evaluator results
            eval_result = evaluator(logits, labels)

            optimizer.zero_grad()
            loss = eval_result['loss']
            acc = eval_result['acc']

            loss.backward()

            max_grad = torch.max(parameters[-1].grad)
            if not torch.isnan(max_grad):
                optimizer.step()
            else:
                print('NAN in gradient, skip this step')
                optimizer.zero_grad()

            # Update Scheduler at this point only if scheduler_type is 'OneCycleLR'
            if scheduler_type == 'OneCycleLR':
                scheduler.step()

            # if idx % 100 == 0:
            #     t_time = time.time()
            #     print(f"Batch:{idx}  loss:{eval_result['loss']}  acc:{eval_result['acc']}   elapsed_time:{t_time - s_time}")
            #     s_time = time.time()

        train_result = evaluator.evalulate_on_cache()
        print(f"Training:   Loss={train_result['loss']},    Acc={train_result['acc']}")

        evaluator.clear_cache()

        ###############################
        # Compute validation error
        ###############################

        model.eval()
        val_itr = iter(val_loader)
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(val_itr):
                input_data = inputs.float().to(device)
                labels = labels.to(device)
                logits = model(input_data)
                eval_result = evaluator(logits, labels)

        val_result = evaluator.evalulate_on_cache()

        print(f"Eval:   Loss={val_result['loss']},    Acc={val_result['acc']}")
        evaluator.clear_cache()

        # Update scheudler here if not 'OneCycleLR'
        if scheduler is not None and scheduler != 'OneCycleLR':
            if scheduler_type == 'reduce_on_plateau':
                #scheduler.step(val_total_err)
                pass
            else:
                scheduler.step()

        # Saving Results:
        if not cfg.DEBUG:
            current_state = {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    }
            if scheduler is not None:
                current_state['scheduler_state'] = scheduler.state_dict()
            torch.save(current_state, state_fpath)

            epoch_result = {
                'epoch': epoch,
                'train_result': train_result,
                'val_result': val_result
            }
            pickle.dump(epoch_result, open(os.path.join(results_path, f"result_epoch_{epoch}.p"), 'wb'))