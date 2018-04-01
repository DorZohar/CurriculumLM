import sys

import general
import main


def train_task_after_curriculum(conf):

    lm_conf = dict(conf)
    lm_conf['task'] = 'LM'

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
