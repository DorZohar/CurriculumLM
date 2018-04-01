import sys

import general
import main


def train_task_after_curriculum(conf):

    lm_conf = dict(conf)
    lm_conf['task'] = 'LM'
    lm_conf['train_steps'] = 34152
    lm_conf['valid_steps'] = 10053
    lm_conf['test_steps'] = 13137

    if not lm_conf['curriculum__input'] and not lm_conf['curriculum__output']:
        model = main.baseline_model(lm_conf)
    else:
        model = main.curriculum_model(lm_conf)

    main.baseline_model(conf, model)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        cfg_path = 'config.py'
    else:
        cfg_path = sys.argv[1]

    conf = general.load_config(cfg_path)

    train_task_after_curriculum(conf)
