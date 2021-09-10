import sys 
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))

import wrappers
import tools
import models
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec
import tensorflow as tf
import numpy as np
import argparse
import collections
import functools
import json
import os
import pathlib
import sys
import time
import soft_actor_critic

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# enable headless training on servers for mujoco
#os.environ['MUJOCO_GL'] = 'egl'

tf.executing_eagerly()

tf.get_logger().setLevel('ERROR')


sys.path.append(str(pathlib.Path(__file__).parent))


def define_config():
    config = tools.AttrDict()
    # General.
    config.logdir = pathlib.Path('.')
    config.seed = 0
    config.steps = 5e6
    config.eval_every = 1e4
    config.log_every = 1e3
    config.log_scalars = True
    config.log_images = True
    config.gpu_growth = True
    config.precision = 16
    # Environment.
    config.task = 'dmc_walker_walk'
    config.envs = 1
    config.difficulty =  'none'
    config.parallel = 'none'
    config.action_repeat = 2
    config.time_limit = 1000
    config.prefill = 5000
    config.eval_noise = 0.0
    config.clip_rewards = 'none'
    # Model.
    config.deter_size = 200
    config.stoch_size = 30
    config.num_units = 400
    config.dense_act = 'elu'
    config.cnn_act = 'relu'
    config.cnn_depth = 32
    config.pcont = False
    config.free_nats = 3.0
    config.kl_scale = 1.0
    config.pcont_scale = 10.0
    config.weight_decay = 0.0
    config.weight_decay_pattern = r'.*'
    # Training.
    config.batch_size = 50
    config.batch_length = 50
    config.train_every = 1000
    config.train_steps = 100
    config.pretrain = 100
    config.model_lr = 6e-4
    config.value_lr = 8e-5
    config.actor_lr = 8e-5
    config.grad_clip = 100.0
    config.dataset_balance = False
    # Behavior.
    config.discount = 0.99
    config.disclam = 0.95
    config.horizon = 15
    config.action_dist = 'tanh_normal'
    config.action_init_std = 5.0
    config.expl = 'additive_gaussian'
    config.expl_amount = 0.3
    config.expl_decay = 0.0
    config.expl_min = 0.0
    config.log_imgs = True

    # natural or not
    config.natural = True
    config.custom_video = True

    # obs model
    config.obs_model = 'generative' # or 'contrastive'



    # use trajectory optimization
    config.trajectory_opt = False
    config.traj_opt_lr = 0.003
    config.num_samples = 20
    return config


class Dreamer(tools.Module):

    def __init__(self, config, datadir, actspace, writer):
        self._c = config
        self._actspace = actspace
        self._actdim = actspace.n if hasattr(
            actspace, 'n') else actspace.shape[0]
        self._writer = writer
        self._random = np.random.RandomState(config.seed)
        with tf.device('cpu:0'):
            self._step = tf.Variable(count_steps(
                datadir, config), dtype=tf.int64)
        self._should_pretrain = tools.Once()
        self._should_train = tools.Every(config.train_every)
        self._should_log = tools.Every(config.log_every)
        self._last_log = None
        self._last_time = time.time()
        self._metrics = collections.defaultdict(tf.metrics.Mean)
        self._metrics['expl_amount']  # Create variable for checkpoint.
        self._float = prec.global_policy().compute_dtype
        self._dataset = iter(load_dataset(datadir, self._c))
        self._build_model()

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step.numpy().item()
        tf.summary.experimental.set_step(step)
        if state is not None and reset.any():
            mask = tf.cast(1 - reset, self._float)[:, None]
            state = tf.nest.map_structure(lambda x: x * mask, state)
        if self._should_train(step):
            log = self._should_log(step)
            n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
            print(f'Training for {n} steps.')
            # with self._strategy.scope():
            for train_step in range(n):
                log_images = self._c.log_images and log and train_step == 0
                self.train(next(self._dataset), log_images)
            if log:
                self._write_summaries()
        action, state = self.policy(obs, state, training)
        if training:
            self._step.assign_add(len(reset) * self._c.action_repeat)
        return action, state

    @tf.function
    def policy(self, obs, state, training):
        if state is None:
            latent = self._dynamics.initial(len(obs['image']))
            action = tf.zeros((len(obs['image']), self._actdim), self._float)
        else:
            latent, action = state
        embed = self._encode(preprocess(obs, self._c))
        latent, _ = self._dynamics.obs_step(latent, action, embed)
        feat = self._dynamics.get_feat(latent)

        if self._c.trajectory_opt:
            action = self._trajectory_optimization(latent)
        else:
            if training:
                action = self._actor(feat).sample()
            else:
                action = self._actor(feat).mode()
 
        action = self._exploration(action, training)
        state = (latent, action)
        return action, state

    def load(self, filename):
        super().load(filename)
        self._should_pretrain()

    @tf.function()
    def train(self, data, log_images=True):
        self._train(data, log_images)

    def _train(self, data, log_images):
        with tf.GradientTape() as model_tape:
            embed = self._encode(data)
            post, prior = self._dynamics.observe(embed, data['action'])
            batch_size = embed.shape[0]
            feat = self._dynamics.get_feat(post)
            reward_pred = self._reward(feat)
            likes = tools.AttrDict()
            likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))

            calc_bs = True
            bs = None

            if calc_bs == True:
             ## for behavioral similarity ##
                from scipy.spatial import distance
                
                try:
                    states_graph = data['position']
                except:
                    states_graph = data['orientations']

                latents_graph = feat 
                assert states_graph.shape[0] == latents_graph.shape[0]

                #K = tf.zeros([batch_size,batch_size],dtype=tf.dtypes.float16)
                c = 1e2
                K = tf.constant([0],dtype=tf.dtypes.float16)
                for i in range(batch_size):
                    for j in range(batch_size):
                        # K[i,j] = max(0,c - abs(distance.cosine(latents_graph[i].numpy(),latents_graph[j].numpy())-distance.cosine(states_graph[i].numpy(),states_graph[j].numpy())) )
                        #tf.norm(x1-y1,ord='euclidean')
                        K = K + (1/(batch_size*batch_size))*tf.math.maximum(tf.constant([0],dtype=tf.dtypes.float16),c - tf.abs(tf.norm(latents_graph[i]-latents_graph[j],ord='euclidean')-tf.norm(states_graph[i]-states_graph[j],ord='euclidean')) )
                        ## compute normalized Kernel distance - a number between 0 and 100. 100 is max similaruuty, 0 is min similarity

                bs = K

                ##  behavioral similarity end  ##

            # if we use the generative observation model, we need to perform observation reconstruction
            image_pred = self._decode(feat)
            # compute the contrative loss directly 
            cont_loss = self._contrastive(feat, embed)

                # the contrastive / generative implementation of the observation model p(o|s)
            if self._c.obs_model == 'generative':
                likes.image = tf.reduce_mean(image_pred.log_prob(data['image']))
            elif self._c.obs_model == 'contrastive':
                likes.image = tf.reduce_mean(cont_loss)

            if self._c.pcont:
                pcont_pred = self._pcont(feat)
                pcont_target = self._c.discount * data['discount']
                likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
                likes.pcont *= self._c.pcont_scale

            prior_dist = self._dynamics.get_dist(prior)
            post_dist = self._dynamics.get_dist(post)
            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            div = tf.maximum(div, self._c.free_nats)
            model_loss = self._c.kl_scale * div - sum(likes.values())

 
        with tf.GradientTape() as actor_tape:
            imag_feat = self._imagine_ahead(post)
            reward = self._reward(imag_feat).mode()
            if self._c.pcont:
                pcont = self._pcont(imag_feat).mean()
            else:
                pcont = self._c.discount * tf.ones_like(reward)
            value = self._value(imag_feat).mode()
            returns = tools.lambda_return(
                reward[:-1], value[:-1], pcont[:-1],
                bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
            discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
                [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
            actor_loss = -tf.reduce_mean(discount * returns)

        with tf.GradientTape() as value_tape:
            value_pred = self._value(imag_feat)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = - \
                tf.reduce_mean(discount * value_pred.log_prob(target))

        actor_norm = self._actor_opt(actor_tape, actor_loss)
        value_norm = self._value_opt(value_tape, value_loss)


        model_norm = self._model_opt(model_tape, model_loss)
        states = tf.concat([post['stoch'], post['deter']], axis=-1)
        rewards = data['reward']
        dones = tf.zeros_like(rewards)
        actions = data['action']


        if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
            if self._c.log_scalars:
                self._scalar_summaries(
                    data, feat, prior_dist, post_dist, likes, div,
                    model_loss, value_loss, actor_loss, model_norm, value_norm,
                    actor_norm,bs)
            if tf.equal(log_images, True) and self._c.log_imgs:
                self._image_summaries(data, embed, image_pred)

    def _build_model(self):
        acts = dict(
            elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
            leaky_relu=tf.nn.leaky_relu)
        cnn_act = acts[self._c.cnn_act]
        act = acts[self._c.dense_act]
        self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act)
        self._dynamics = models.RSSM(
            self._c.stoch_size, self._c.deter_size, self._c.deter_size)
        self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act)
        self._contrastive = models.ContrastiveObsModel(self._c.deter_size,
                self._c.deter_size * 2)
        self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
        if self._c.pcont:
            self._pcont = models.DenseDecoder(
                (), 3, self._c.num_units, 'binary', act=act)
        self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
        self._Qs = [models.QNetwork(3, self._c.num_units, act=act) for _ in range(self._c.num_Qs)]
        self._actor = models.ActionDecoder(
            self._actdim, 4, self._c.num_units, self._c.action_dist,
            init_std=self._c.action_init_std, act=act)
        model_modules = [self._encode, self._dynamics,
                         self._contrastive, self._reward, self._decode]
        if self._c.pcont:
            model_modules.append(self._pcont)
        Optimizer = functools.partial(
            tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
            wdpattern=self._c.weight_decay_pattern)
        self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
        self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
        self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
        self._q_opts = [Optimizer('qs', [qnet], self._c.value_lr) for qnet in self._Qs]



        self.train(next(self._dataset))

    def _exploration(self, action, training):
        if training:
            amount = self._c.expl_amount
            if self._c.expl_decay:
                amount *= 0.5 ** (tf.cast(self._step,
                                          tf.float32) / self._c.expl_decay)
            if self._c.expl_min:
                amount = tf.maximum(self._c.expl_min, amount)
            self._metrics['expl_amount'].update_state(amount)
        elif self._c.eval_noise:
            amount = self._c.eval_noise
        else:
            return action
        if self._c.expl == 'additive_gaussian':
            return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
        if self._c.expl == 'completely_random':
            return tf.random.uniform(action.shape, -1, 1)
        if self._c.expl == 'epsilon_greedy':
            indices = tfd.Categorical(0 * action).sample()
            return tf.where(
                tf.random.uniform(action.shape[:1], 0, 1) < amount,
                tf.one_hot(indices, action.shape[-1], dtype=self._float),
                action)
        raise NotImplementedError(self._c.expl)

    def _imagine_ahead(self, post):
        if self._c.pcont:  # Last step could be terminal.
            post = {k: v[:, :-1] for k, v in post.items()}

        def flatten(x): return tf.reshape(x, [-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in post.items()}

        def policy(state): return self._actor(
            tf.stop_gradient(self._dynamics.get_feat(state))).sample()
        states = tools.static_scan(
            lambda prev, _: self._dynamics.img_step(prev, policy(prev)),
            tf.range(self._c.horizon), start)
        imag_feat = self._dynamics.get_feat(states)
        return imag_feat

    def _trajectory_optimization(self, post):
        def policy(state): return self._actor(
            tf.stop_gradient(self._dynamics.get_feat(state))).sample()

        def repeat(x):
            return tf.repeat(x, self._c.num_samples, axis=0)

        states, actions = tools.static_scan_action(
            lambda prev, action,  _: self._dynamics.img_step(prev, action),
            lambda prev: policy(prev),
            tf.range(self._c.horizon), post)

        feat = self._dynamics.get_feat(states)
        reward = self._reward(feat).mode()

        if self._c.pcont:
            pcont = self._pcont(feat).mean()
        else:
            pcont = self._c.discount * tf.ones_like(reward)
        value = self._value(feat).mode()

        # compute the accumulated reward
        returns = tools.lambda_return(
            reward[:-1], value[:-1], pcont[:-1],
            bootstrap=value[-1], lambda_=self._c.disclam, axis=0)

        accumulated_reward = returns[0, 0]

        # since the reward and latent dynamics are fully differentiable, we can backprop the gradients to update the actions
        grad = tf.gradients(accumulated_reward, actions)[0]
        act = actions + grad * self._c.traj_opt_lr

        return act


    def _scalar_summaries(
            self, data, feat, prior_dist, post_dist, likes, div,
            model_loss, value_loss, actor_loss, model_norm, value_norm,
            actor_norm,bs=None):
        self._metrics['model_grad_norm'].update_state(model_norm)
        self._metrics['value_grad_norm'].update_state(value_norm)
        self._metrics['actor_grad_norm'].update_state(actor_norm)
        self._metrics['prior_ent'].update_state(prior_dist.entropy())
        self._metrics['post_ent'].update_state(post_dist.entropy())
        for name, logprob in likes.items():
            self._metrics[name + '_loss'].update_state(-logprob)
        self._metrics['div'].update_state(div)
        self._metrics['model_loss'].update_state(model_loss)
        self._metrics['value_loss'].update_state(value_loss)
        self._metrics['actor_loss'].update_state(actor_loss)
        self._metrics['action_ent'].update_state(self._actor(feat).entropy())

        if bs is not None:
            self._metrics['bs'].update_state(bs)        

    def _image_summaries(self, data, embed, image_pred):
        truth = data['image'][:6] + 0.5
        recon = image_pred.mode()[:6]
        init, _ = self._dynamics.observe(embed[:6, :5], data['action'][:6, :5])
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self._dynamics.imagine(data['action'][:6, 5:], init)
        openl = self._decode(self._dynamics.get_feat(prior)).mode()
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        openl = tf.concat([truth, model, error], 2)
        tools.graph_summary(
            self._writer, tools.video_summary, 'agent/openl', openl)

    def _write_summaries(self):
        step = int(self._step.numpy())
        metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
        if self._last_log is not None:
            duration = time.time() - self._last_time
            self._last_time += duration
            metrics.append(('fps', (step - self._last_log) / duration))
        self._last_log = step
        [m.reset_states() for m in self._metrics.values()]
        with (self._c.logdir / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
        [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
        print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
        self._writer.flush()


def preprocess(obs, config):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    with tf.device('cpu:0'):
        obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
        clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[
            config.clip_rewards]
        obs['reward'] = clip_rewards(obs['reward'])
    return obs


def count_steps(datadir, config):
    return tools.count_episodes(datadir)[1] * config.action_repeat


def load_dataset(directory, config):
    episode = next(tools.load_episodes(directory, 1))
    types = {k: v.dtype for k, v in episode.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}

    def generator(): return tools.load_episodes(
        directory, config.train_steps, config.batch_length,
        config.dataset_balance)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.map(functools.partial(preprocess, config=config))
    dataset = dataset.prefetch(10)
    return dataset


def summarize_episode(episode, config, datadir, writer, prefix):
    episodes, steps = tools.count_episodes(datadir)
    length = (len(episode['reward']) - 1) * config.action_repeat
    ret = episode['reward'].sum()
    print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
    metrics = [
        (f'{prefix}/return', float(episode['reward'].sum())),
        (f'{prefix}/length', len(episode['reward']) - 1),
        (f'episodes', episodes)]
    step = count_steps(datadir, config)
    with (config.logdir / 'metrics.jsonl').open('a') as f:
        f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
    with writer.as_default():  # Env might run in a different thread.
        tf.summary.experimental.set_step(step)
        [tf.summary.scalar('sim/' + k, v) for k, v in metrics]
        if prefix == 'test':
            tools.video_summary(f'sim/{prefix}/video', episode['image'][None])


def make_env(config, writer, prefix, datadir, train):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
        if config.difficulty == 'none':
            env = wrappers.DeepMindControl(task)
        else:
            env = wrappers.DeepMindControlDistraction(task,difficulty=config.difficulty)
        env = wrappers.ActionRepeat(env, config.action_repeat)
        env = wrappers.NormalizeActions(env)
        if config.natural:
            data = load_imgnet(train)
            env = wrappers.NaturalMujoco(env, data)
        elif config.custom_video:
            import pickle
            
            with oen('custom_video_jaco.pkl', 'rb') as file:
                data = pickle.load(file)
            env = wrappers.CustomMujoco(env, data)
    elif suite == 'atari':
        env = wrappers.Atari(
            task, config.action_repeat, (64, 64), grayscale=False,
            life_done=True, sticky_actions=True)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
    callbacks = []
    if train:
        callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
    callbacks.append(
        lambda ep: summarize_episode(ep, config, datadir, writer, prefix))
    env = wrappers.Collect(env, callbacks, config.precision)
    env = wrappers.RewardObs(env)
    return env
def make_env_test(config, writer, prefix, datadir, train):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
        if config.difficulty == 'none':
            env = wrappers.DeepMindControl(task)
        else:
            env = wrappers.DeepMindControlDistraction(task,difficulty=config.difficulty)
        env = wrappers.ActionRepeat(env, config.action_repeat)
        env = wrappers.NormalizeActions(env)
        if config.natural:
            data = load_imgnet(train)
            env = wrappers.NaturalMujoco(env, data)
        elif config.custom_video:
            import pickle
            
            with gfile.Open('custom_video_jaco.pkl', 'rb') as file:
                data = pickle.load(file)
            env = wrappers.CustomMujoco(env, data)
    elif suite == 'atari':
        env = wrappers.Atari(
            task, config.action_repeat, (64, 64), grayscale=False,
            life_done=True, sticky_actions=True)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
    callbacks = []
    if train:
        callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
    callbacks.append(
        lambda ep: summarize_episode(ep, config, datadir, writer, prefix))
    env = wrappers.Collect(env, callbacks, config.precision)
    env = wrappers.RewardObs(env)
    return env

def load_imgnet(train):
    import pickle
    name = 'train' if train else 'valid'

    # images_train. pkl and images_test.pkl to be downloaded from 

    with open('images_{}.pkl'.format(name), 'rb') as fin:
        imgnet = pickle.load(fin)

    imgnet = np.transpose(imgnet, axes=(0, 1, 3, 4, 2))

    return imgnet


def main(config):
    print('mainn')
    if config.gpu_growth:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        prec.set_policy(prec.Policy('mixed_float16'))
    config.steps = int(config.steps)
    config.logdir.mkdir(parents=True, exist_ok=True)
    print('Logdir', config.logdir)

    arg_dict = vars(config).copy()
    del arg_dict['logdir']

    # with open(os.path.join(config.logdir, 'args.json'), 'w') as fout:
    #     import json
    #     json.dump(arg_dict, fout)

    # Create environments.
    datadir = config.logdir / 'episodes'
    datadir.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(
        str(config.logdir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    train_envs = [wrappers.Async(lambda: make_env(
        config, writer, 'train', datadir, train=True), config.parallel)
        for _ in range(config.envs)]
    test_envs = [wrappers.Async(lambda: make_env_test(
        config, writer, 'test', datadir, train=False), config.parallel)
        for _ in range(config.envs)]
    actspace = train_envs[0].action_space

    # Prefill dataset with random episodes.
    step = count_steps(datadir, config)
    prefill = max(0, config.prefill - step)
    print(f'Prefill dataset with {prefill} steps.')
    def random_agent(o, d, _): return ([actspace.sample() for _ in d], None)
    tools.simulate(random_agent, train_envs, prefill / config.action_repeat)
    writer.flush()

    # Train and regularly evaluate the agent.
    step = count_steps(datadir, config)
    print(f'Simulating agent for {config.steps-step} steps.')
    agent = Dreamer(config, datadir, actspace, writer)
    if (config.logdir / 'variables.pkl').exists():
        print('Load checkpoint.')
        agent.load(config.logdir / 'variables.pkl')
    state = None
    while step < config.steps:
        print('Start evaluation.')
        tools.simulate(
            functools.partial(agent, training=False), test_envs, episodes=1)
        writer.flush()
        print('Start collection.')
        steps = config.eval_every // config.action_repeat
        state = tools.simulate(agent, train_envs, steps, state=state)
        step = count_steps(datadir, config)
        agent.save(config.logdir / 'variables.pkl')
    for env in train_envs + test_envs:
        env.close()


#if __name__ == '__main__':
    # try:
    #     import colored_traceback
    #     colored_traceback.add_hook()
    # except ImportError:
    #     pass
parser = argparse.ArgumentParser()
for key, value in define_config().items():
    parser.add_argument(
        f'--{key}', type=tools.args_type(value), default=value)
args = parser.parse_args()

print('main')

main(args)
