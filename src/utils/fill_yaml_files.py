import yaml
import os,copy
import argparse


EXPERIMENTS_DIRECTORY = "src/config/experiments"
EXPERIMENTS_FILE_EXTENSION = ".yaml"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Experiments')
    parser.add_argument('--experiment','-e',help=f"Experiment config file name from '{EXPERIMENTS_DIRECTORY}' ")
    parser.add_argument('--pretrainedpath','-p',help=f"Pretrained Model Path")
    parser.add_argument('--alphabet','-a',help=f"Alphabet of Omniglot to train on")
    experiment = parser.parse_args().experiment

    filepath = os.path.abspath(f'{EXPERIMENTS_DIRECTORY}/{experiment}{EXPERIMENTS_FILE_EXTENSION}')

    with open(filepath,'r') as yamlfile:
        cur_yaml = yaml.safe_load(yamlfile) # Note the safe_load

    output_dims = []
    single_omniglot_classes = None
    DATASET_PATH = 'src/data/datasets/omniglot/Omniglot/'
    tasks = sorted(list(filter(lambda x: x[0]!='.', os.listdir(os.path.abspath(DATASET_PATH)))))
    for task_name in tasks:
        num_chars = len( list(filter(lambda x: x[0]!='.', os.listdir(DATASET_PATH+task_name))) )
        output_dims.append(num_chars)
        if task_name == parser.parse_args().alphabet:
            single_omniglot_classes = num_chars

    if experiment[:8] == 'finetune':
        cur_yaml['MODEL']['PRETRAINED_PATH'] = parser.parse_args().pretrainedpath

    if experiment == 'omniglot_single_task':
        cur_yaml['DATASET']['ALPHABET'] = parser.parse_args().alphabet
        cur_yaml['MODEL']['BODY']['OUTPUT_DIM'] = single_omniglot_classes
        cur_yaml['MODEL']['HEADS']['OUTPUT_DIMS'] = [single_omniglot_classes]

    if experiment == 'multitask1':
        cur_yaml['MODEL']['BODY']['HIDDEN_DIMS'] = [1024]
        cur_yaml['MODEL']['BODY']['OUTPUT_DIM'] = 1024
        cur_yaml['MODEL']['HEADS']['NAMES'] = ['simple_head' for _ in range(30)]
        cur_yaml['MODEL']['HEADS']['INPUT_DIMS'] = [1024 for _ in range(30)]
        cur_yaml['MODEL']['HEADS']['HIDDEN_DIMS'] = [ [] for _ in range(30)]
        cur_yaml['MODEL']['HEADS']['HIDDEN_DIMS_FREEZE'] = [ [False] for _ in range(30)]
        cur_yaml['MODEL']['HEADS']['OUTPUT_DIMS'] = output_dims.copy()
    if experiment == 'multitask2':
        cur_yaml['MODEL']['BODY']['HIDDEN_DIMS'] = []
        cur_yaml['MODEL']['BODY']['OUTPUT_DIM'] = 1024
        cur_yaml['MODEL']['HEADS']['NAMES'] = ['simple_head' for _ in range(30)]
        cur_yaml['MODEL']['HEADS']['INPUT_DIMS'] = [1024 for _ in range(30)]
        cur_yaml['MODEL']['HEADS']['HIDDEN_DIMS'] = [ [1024] for head_idx in range(30)]
        cur_yaml['MODEL']['HEADS']['HIDDEN_DIMS_FREEZE'] = [ [False] for _ in range(30)]
        cur_yaml['MODEL']['HEADS']['OUTPUT_DIMS'] = output_dims.copy()


    if cur_yaml:
        with open(filepath,'w') as yamlfile:
            yaml.safe_dump(cur_yaml, yamlfile, sort_keys=False) # Also note the safe_dump
