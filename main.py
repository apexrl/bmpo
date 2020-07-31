import argparse
import importlib
import runner
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
# parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

module = importlib.import_module(args.config)
params = getattr(module, 'params')
universe, domain, task = params['universe'], params['domain'], params['task']
epoch_length = params['kwargs']['epoch_length']

NUM_EPOCHS_PER_DOMAIN = {
    'Pendulum':int(19),
    'Hopper': int(95),
    'HopperNT': int(95),
    'Walker2d': int(195),
    'Walker2dNT': int(195),
    'Ant': int(295),
}


params['kwargs']['n_epochs'] = NUM_EPOCHS_PER_DOMAIN[domain]
#params['kwargs']['n_initial_exploration_steps'] = 5000
params['kwargs']['reparameterize'] = True
params['kwargs']['lr'] = 3e-4
params['kwargs']['target_update_interval'] = 1
params['kwargs']['tau'] = 5e-3
params['kwargs']['store_extra_policy_info'] = False
params['kwargs']['action_prior'] = 'uniform'

variant_spec = {
        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': {},
            },
            'evaluation': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': {},
            },
        },
        'policy_params': {
            'type': 'GaussianPolicy',
            'kwargs': {
                'hidden_layer_sizes': (256, 256),
                'squash': True,
            }
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (256, 256),
            }
        },
        'algorithm_params': params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': int(1e6),
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': epoch_length,
                'min_pool_size': epoch_length,
                'batch_size': 256,
            }
        },
        'run_params': {
            # 'seed': args.seed,
            'checkpoint_at_end': True,
            'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN[domain] // 10,
            'checkpoint_replay_pool': False,
        },
    }

exp_runner = runner.ExperimentRunner(variant_spec)
diagnostics = exp_runner.train()
