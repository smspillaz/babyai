from abc import ABC, abstractmethod
import torch
from .. import utils
from babyai.bot import Bot
from babyai.model import ACModel
from random import Random
import numpy as np


class Agent(ABC):
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def on_reset(self):
        pass

    @abstractmethod
    def act(self, obs):
        """Propose an action based on observation.

        Returns a dict, with 'action` entry containing the proposed action,
        and optionaly other entries containing auxiliary information
        (e.g. value function).

        """
        pass

    @abstractmethod
    def analyze_feedback(self, reward, done):
        pass


class ModelAgent(Agent):
    """A model-based agent. This agent behaves using a model."""

    def __init__(self, model_or_name, obss_preprocessor, argmax, split_model=None):
        if obss_preprocessor is None:
            assert isinstance(model_or_name, str)
            obss_preprocessor = utils.ObssPreprocessor(model_or_name)
        self.obss_preprocessor = obss_preprocessor
        if isinstance(model_or_name, str):
            self.model = utils.load_model(model_or_name)
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            self.model = model_or_name
        self.device = next(self.model.parameters()).device
        self.argmax = argmax
        self.memory = None
        self.sentence_segments = None

        # Load the sentence splitting model
        if isinstance(split_model, str):
            self.split_model, self.split_idxs = utils.load_split_model(split_model)
        else:
            self.split_model, self.split_idxs = (None, None)

    def act_batch(self, many_obs):
        if self.sentence_segments is None:
            self.sentence_segments = np.zeros(len(many_obs), dtype=int)

        if self.memory is None:
            self.memory = torch.zeros(
                len(many_obs), self.model.memory_size, device=self.device)
        elif self.memory.shape[0] != len(many_obs):
            raise ValueError("stick to one batch size for the lifetime of an agent")

        # We are segmenting the sentence every time. Its not ideal, but
        # this is the least invasive way of doing it.
        if self.split_model is not None:
            segments = [
                " ".join(list(utils.split_sentence_by_model(self.split_model, self.split_idxs, o["mission"]))[0])
                for o in many_obs
            ]
        else:
            segments = [[o["mission"]] for o in many_obs]

        new_many_obs = [
            { **o, "mission": segments[i][self.sentence_segments[i]] }
            for i, o in enumerate(many_obs)
        ]

        preprocessed_obs = self.obss_preprocessor(new_many_obs, device=self.device)

        with torch.no_grad():
            model_results = self.model(preprocessed_obs, self.memory)
            dist = model_results['dist']
            value = model_results['value']
            self.memory = model_results['memory']

        if self.argmax:
            action = dist.probs.argmax(1)
        else:
            action = dist.sample()

        # If an agent picks "done" we must move on to the next instruction
        # in the segmentation
        for i, act in enumerate(action):
            if act == dist.logits.shape[-1] - 1: # assuming that this is the done action
                self.sentence_segments[i] = min(self.sentence_segments[i] + 1, len(segments[i]) - 1)
                while len(segments[i]) == 1:
                    self.sentence_segments[i] = min(self.sentence_segments[i] + 1, len(segments))
                print('done', i, 'next', self.sentence_segments[i])

        return {'action': action,
                'dist': dist,
                'value': value}

    def act(self, obs):
        return self.act_batch([obs])

    def analyze_feedback(self, reward, done):
        if isinstance(done, tuple):
            for i in range(len(done)):
                if done[i]:
                    self.memory[i, :] *= 0
                    self.sentence_segments[i] = 0
        else:
            self.memory *= (1 - done)


class RandomAgent:
    """A newly initialized model-based agent."""

    def __init__(self, seed=0, number_of_actions=7):
        self.rng = Random(seed)
        self.number_of_actions = number_of_actions

    def act(self, obs):
        action = self.rng.randint(0, self.number_of_actions - 1)
        # To be consistent with how a ModelAgent's output of `act`:
        return {'action': torch.tensor(action),
                'dist': None,
                'value': None}


class DemoAgent(Agent):
    """A demonstration-based agent. This agent behaves using demonstrations."""

    def __init__(self, demos_name, env_name, origin):
        self.demos_path = utils.get_demos_path(demos_name, env_name, origin, valid=False)
        self.demos = utils.load_demos(self.demos_path)
        self.demos = utils.demos.transform_demos(self.demos)
        self.demo_id = 0
        self.step_id = 0

    @staticmethod
    def check_obss_equality(obs1, obs2):
        if not(obs1.keys() == obs2.keys()):
            return False
        for key in obs1.keys():
            if type(obs1[key]) in (str, int):
                if not(obs1[key] == obs2[key]):
                    return False
            else:
                if not (obs1[key] == obs2[key]).all():
                    return False
        return True

    def act(self, obs):
        if self.demo_id >= len(self.demos):
            raise ValueError("No demonstration remaining")
        expected_obs = self.demos[self.demo_id][self.step_id][0]
        assert DemoAgent.check_obss_equality(obs, expected_obs), "The observations do not match"

        return {'action': self.demos[self.demo_id][self.step_id][1]}

    def analyze_feedback(self, reward, done):
        self.step_id += 1

        if done:
            self.demo_id += 1
            self.step_id = 0


class BotAgent:
    def __init__(self, env):
        """An agent based on a GOFAI bot."""
        self.env = env
        self.on_reset()

    def on_reset(self):
        self.bot = Bot(self.env)

    def act(self, obs=None, update_internal_state=True, *args, **kwargs):
        action = self.bot.replan()
        return {'action': action}

    def analyze_feedback(self, reward, done):
        pass


def load_agent(env, model_name, demos_name=None, demos_origin=None, argmax=True, env_name=None, split_model=None):
    # env_name needs to be specified for demo agents
    if model_name == 'BOT':
        return BotAgent(env)
    elif model_name is not None:
        obss_preprocessor = utils.ObssPreprocessor(model_name, env.observation_space)
        return ModelAgent(model_name, obss_preprocessor, argmax, split_model=split_model)
    elif demos_origin is not None or demos_name is not None:
        return DemoAgent(demos_name=demos_name, env_name=env_name, origin=demos_origin)
