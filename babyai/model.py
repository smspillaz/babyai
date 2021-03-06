import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import babyai.rl
from babyai.rl.utils.supervised_losses import required_heads


# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=imm_channels,
            kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features,
            kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))


class ImageBOWEmbedding(nn.Module):
   def __init__(self, max_value, n_channels, embedding_dim):
       super().__init__()
       self.max_value = max_value
       self.embedding_dim = embedding_dim
       self.embedding = nn.Embedding(n_channels * max_value, embedding_dim)
       self.apply(initialize_parameters)

   def forward(self, inputs):
       offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(inputs.device)
       inputs = (inputs + offsets[None, :, None, None]).long()
       return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)


class ImageEncoder(nn.Module):
    def __init__(self,
                 obs_space,
                 with_manager_map=False,
                 image_dim=128,
                 arch="bow_endpool_res"):
        super().__init__()
        endpool = 'endpool' in arch
        use_bow = 'bow' in arch
        pixel = 'pixel' in arch
        self.res = 'res' in arch
        self.with_manager_map = with_manager_map

        self.arch = arch

        if self.res and image_dim != 128:
            raise ValueError(f"image_dim is {image_dim}, expected 128")
        self.image_dim = image_dim

        for part in self.arch.split('_'):
            if part not in ['original', 'bow', 'pixels', 'endpool', 'res']:
                raise ValueError("Incorrect architecture name: {}".format(self.arch))

        self.image_conv = nn.Sequential(*[
            *([ImageBOWEmbedding(obs_space['image'], 4 if with_manager_map else 3, 128)] if use_bow else []),
            *([nn.Conv2d(
                in_channels=4 if with_manager_map else 3, out_channels=128, kernel_size=(8, 8),
                stride=8, padding=0)] if pixel else []),
            nn.Conv2d(
                in_channels=128 if use_bow or pixel else (4 if with_manager_map else 3), out_channels=128,
                kernel_size=(3, 3) if endpool else (2, 2), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
        ])

    def forward(self, x, manager_map=None):
        if self.with_manager_map:
            torch.cat([x, manager_map.unsqueeze(-1)], dim=-1)

        x = torch.transpose(torch.transpose(x, 1, 3), 2, 3)

        if 'pixel' in self.arch:
            x /= 256.0
        x = self.image_conv(x)

        return x


class LanguageEncoder(nn.Module):
    def __init__(self,
                 obs_space,
                 instr_dim,
                 lang_model="gru",
                 arch="bow_endpool_res"):
        super().__init__()
        self.instr_dim = instr_dim
        self.arch = arch
        self.lang_model = lang_model
        self.obs_space = obs_space

        for part in self.arch.split('_'):
            if part not in ['original', 'bow', 'pixels', 'endpool', 'res']:
                raise ValueError("Incorrect architecture name: {}".format(self.arch))

        if self.lang_model in ['gru', 'bigru', 'attgru']:
            self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
            if self.lang_model in ['gru', 'bigru', 'attgru']:
                gru_dim = self.instr_dim
                if self.lang_model in ['bigru', 'attgru']:
                    gru_dim //= 2
                self.instr_rnn = nn.GRU(
                    self.instr_dim, gru_dim, batch_first=True,
                    bidirectional=(self.lang_model in ['bigru', 'attgru']))
                self.final_instr_dim = self.instr_dim
            else:
                kernel_dim = 64
                kernel_sizes = [3, 4]
                self.instr_convs = nn.ModuleList([
                    nn.Conv2d(1, kernel_dim, (K, self.instr_dim)) for K in kernel_sizes])
                self.final_instr_dim = kernel_dim * len(kernel_sizes)

    def forward(self, instr):
        lengths = (instr != 0).sum(1).long()
        if self.lang_model == 'gru':
            out, _ = self.instr_rnn(self.word_embedding(instr))
            hidden = out[range(len(lengths)), lengths-1, :]
            return hidden

        elif self.lang_model in ['bigru', 'attgru']:
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]
                outputs, final_states = self.instr_rnn(self.word_embedding(instr))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            return outputs if self.lang_model == 'attgru' else final_states

        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))


class FiLMPooling(nn.Module):
    def __init__(self,
                 arch="bow_endpool_res"):
        super().__init__()

        endpool = 'endpool' in arch
        self.film_pool = nn.MaxPool2d(kernel_size=(7, 7) if endpool else (2, 2), stride=2)

    def forward(self, x, *args, **kwargs):
        x = F.relu(self.film_pool(x))
        x = x.reshape(x.shape[0], -1)

        return x


class FiLMImageConditioning(nn.Module):
    def __init__(self,
                 image_dim,
                 instr_dim,
                 arch="bow_endpool_res"):
        super().__init__()
        self.arch = arch

        endpool = 'endpool' in arch
        use_bow = 'bow' in arch
        pixel = 'pixel' in arch
        self.res = 'res' in arch

        for part in self.arch.split('_'):
            if part not in ['original', 'bow', 'pixels', 'endpool', 'res']:
                raise ValueError("Incorrect architecture name: {}".format(self.arch))

        num_module = 2
        self.controllers = []
        for ni in range(num_module):
            mod = FiLM(
                in_features=instr_dim,
                out_features=128 if ni < num_module-1 else image_dim,
                in_channels=128, imm_channels=128)
            self.controllers.append(mod)
            self.add_module('FiLM_' + str(ni), mod)

    def forward(self, img_embedding, instr_embedding):
        x = img_embedding

        if instr_embedding is not None:
            for controller in self.controllers:
                out = controller(x, instr_embedding)
                if self.res:
                    out += x
                x = out

        return x


class MemoryLanguageAttention(nn.Module):
    def __init__(self,
                 instr_dim,
                 memory_dim=128,
                 arch="bow_endpool_res",
                 lang_model="gru"):
        super().__init__()

        endpool = 'endpool' in arch
        use_bow = 'bow' in arch
        pixel = 'pixel' in arch
        self.res = 'res' in arch

        self.arch = arch
        self.lang_model = lang_model
        self.memory_dim = memory_dim

        if self.lang_model == 'attgru':
            self.memory2key = nn.Linear(self.memory_size, instr_dim)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, instr, instr_embedding, memory):
        if self.lang_model == "attgru":
            # outputs: B x L x D
            # memory: B x M
            mask = (instr != 0).float()
            # The mask tensor has the same length as obs.instr, and
            # thus can be both shorter and longer than instr_embedding.
            # It can be longer if instr_embedding is computed
            # for a subbatch of obs.instr.
            # It can be shorter if obs.instr is a subbatch of
            # the batch that instr_embeddings was computed for.
            # Here, we make sure that mask and instr_embeddings
            # have equal length along dimension 1.
            mask = mask[:, :instr_embedding.shape[1]]
            instr_embedding = instr_embedding[:, :mask.shape[1]]

            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        return instr_embedding


class Memory(nn.Module):
    def __init__(self,
                 image_dim,
                 memory_dim=128):
        super().__init__()
        self.image_dim = image_dim
        self.memory_dim = memory_dim

        self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)

    def forward(self, x, memory):
        hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        hidden = self.memory_rnn(x, hidden)
        embedding = hidden[0]
        memory = torch.cat(hidden, dim=1)

        return embedding, memory

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim


class StateEncoder(nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 image_dim=128,
                 memory_dim=128,
                 instr_dim=128,
                 use_instr=False,
                 lang_model="gru",
                 use_memory=False,
                 arch="bow_endpool_res"):
        super().__init__()

        self.use_instr = use_instr
        self.use_memory = use_memory
        self.image_dim = image_dim

        self.image_encoder = ImageEncoder(
            obs_space,
            image_dim=image_dim,
            arch=arch
        )

        # Define instruction embedding
        if self.use_instr:
            self.language_encoder = LanguageEncoder(
                obs_space,
                instr_dim,
                lang_model=lang_model,
                arch=arch
            )

            self.memory_language_attention = MemoryLanguageAttention(
                instr_dim,
                memory_dim=memory_dim,
                lang_model=lang_model,
                arch=arch
            )

            self.film_image_conditioning = FiLMImageConditioning(
                self.image_dim,
                self.language_encoder.final_instr_dim,
                arch=arch
            )

        self.film_pooling = FiLMPooling(arch=arch)

        if self.use_memory:
            self.memory = Memory(
                image_dim,
                memory_dim=memory_dim
            )
            self.embedding_size = self.memory.semi_memory_size
        else:
            self.embedding_size = image_dim

    @property
    def memory_size(self):
        return self.memory.memory_size

    @property
    def semi_memory_size(self):
        return self.memory.semi_memory_size

    def forward(self, obs, memory, instr_embedding):
        x = self.image_encoder(obs.image)

        if self.use_instr and instr_embedding is None:
            instr_embedding = self.language_encoder(obs.instr)

        if self.use_instr:
            instr_embedding = self.memory_language_attention(
                obs.instr,
                instr_embedding,
                memory
            )

            x = self.film_image_conditioning(x, instr_embedding if self.use_instr else None)

        x = self.film_pooling(x)

        if self.use_memory:
            embedding, memory = self.memory(x, memory)
        else:
            embedding = x
        
        return embedding, memory


class ExtraHeads(nn.Module):
    def __init__(self, embedding_size, aux_info=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.aux_info = aux_info

    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    def forward(self, embedding):
        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        return extra_predictions


class ACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False,
                 arch="bow_endpool_res", aux_info=None, **kwargs):
        super().__init__()
    
        self.observation_latent_size = 1

        self.state_encoder = StateEncoder(obs_space,
                                          action_space,
                                          image_dim,
                                          memory_dim,
                                          instr_dim,
                                          use_instr,
                                          lang_model,
                                          use_memory,
                                          arch)

        # Define memory and resize image embedding
        self.embedding_size = self.state_encoder.embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        self.extra_heads = ExtraHeads(self.embedding_size, aux_info)

    def add_heads(self):
        self.extra_heads.add_heads()

    def add_extra_heads_if_necessary(self, aux_info):
        self.extra_heads.add_extra_heads_if_necessary(self, aux_info)

    @property
    def memory_size(self):
        return self.state_encoder.memory_size

    @property
    def semi_memory_size(self):
        return self.state_encoder.semi_memory_size

    def forward(self, obs, memory, instr_embedding=None, **kwargs):
        embedding, memory = self.state_encoder(obs, memory, instr_embedding)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        extra_predictions = self.extra_heads(embedding)

        return {
            'dist': dist,
            'value': value,
            'memory': memory,
            'manager_dist': None,
            'manager_observation_probs': None,
            'extra_predictions': extra_predictions
        }


class HierarchicalACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False,
                 arch="bow_endpool_res", aux_info=None, p_range=(5, 15), latent_size=16):
        """Implements HiPPO within the ACModel.

        The only crucial difference here is that we have a "manager" policy
        which re-generates a latent every p timesteps, where p is a uniformly
        distributed random variable. Then the actor policy is conditioned on
        that latent.
        """
        super().__init__()
        self.state_encoder = StateEncoder(obs_space,
                                          action_space,
                                          image_dim,
                                          memory_dim,
                                          instr_dim,
                                          use_instr,
                                          lang_model,
                                          use_memory,
                                          arch)

        # Define memory and resize image embedding
        self.embedding_size = image_dim
        self.latent_size = latent_size
        self.action_space = action_space
        self.countdown = 0

        # Define the manager model
        self.manager = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.latent_size)
        )

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size + self.latent_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        self.extra_heads = ExtraHeads(self.embedding_size, aux_info)

    def add_heads(self):
        self.extra_heads.add_heads()

    def add_extra_heads_if_necessary(self, aux_info):
        self.extra_heads.add_extra_heads_if_necessary(self, aux_info)

    @property
    def memory_size(self):
        return self.state_encoder.memory_size

    @property
    def semi_memory_size(self):
        return self.state_encoder.semi_memory_size

    def forward(self, obs, memory, instr_embedding=None, manager_latent=None):
        manager_latent = (
            torch.zeros(
                memory.shape[0], self.latent_size, device=memory.device
            )
            if manager_latent is None else manager_latent
        )
        embedding, memory = self.state_encoder(obs, memory, instr_embedding)

        x = self.actor(torch.cat([embedding, manager_latent], dim=-1))
        dist = Categorical(logits=F.log_softmax(x, dim=-1))

        x = self.manager(embedding)
        manager_dist = Categorical(logits=F.log_softmax(x, dim=-1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        extra_predictions = self.extra_heads(embedding)

        return {
            'dist': dist,
            'manager_dist': manager_dist,
            'value': value,
            'memory': memory,
            'extra_predictions': extra_predictions
        }


class Heatmap(nn.Module):
    def __init__(self,
                 embedding_size,
                 in_channels):
        super().__init__()
        self.film_image_conditioning = FiLMImageConditioning(
            in_channels,
            embedding_size
        )

    def forward(self, x, latent):
        x = self.film_image_conditioning(x, latent)

        b, c, h, w = x.shape

        x = F.softmax(x.view(b, c, -1), dim=-1).view(b, c, h, w)

        return x


class ManagerObservations(nn.Module):
    def __init__(self,
                 embedding_size,
                 instr_embedding_size,
                 manager_observation_size,
                 num_manager_observations):
        super().__init__()
        self.embedding_size = embedding_size
        self.instr_embedding_size = instr_embedding_size
        self.nets = [
            nn.Sequential(
                nn.Linear(self.embedding_size + self.instr_embedding_size, manager_observation_size),
                nn.Tanh(),
                nn.Linear(manager_observation_size, manager_observation_size)
            )
            for i in range(num_manager_observations)
        ]

    def forward(self, embedding, instr_embedding):
        x = torch.cat([embedding, instr_embedding], dim=-1)
        x = torch.stack([
            head(x)
            for head in self.nets
        ], dim=1)

        return x


def rotate_image_batch(images, directions):
    return torch.stack([
        torch.rot90(image, k)
        for image, k in zip(images, directions)
    ])


def make_observation_masks(manager_action_indicator, action_indicator_dim, directions, batch_size, image_height, image_width, device=None):
    masks = (
        torch.nn.functional.one_hot(manager_action_indicator, action_indicator_dim).to(torch.float)
        if manager_action_indicator is not None
        else torch.zeros(batch_size, action_indicator_dim, device=device, dtype=torch.float)
    ).reshape(batch_size, image_height, image_width)

    # Rotate counterclockwise back to agent view
    masks = rotate_image_batch(masks, -1 * directions)

    return masks



class LanguageConditionedHierarchicalACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False,
                 arch="bow_endpool_res", aux_info=None, p_range=(5, 15),
                 action_latent_size=16,
                 observation_latent_size=16,
                 observation_latents=8):
        """Implements HiPPO within the ACModel, using the language with the manager.

        Here the manager gets the language and the lower level policy doesn't. The manager
        produces some latent which the worker is supposed to follow.

        Note here that we can't use the same abstraction for the manager and the worker
        they get different inputs by design.
        """
        super().__init__()
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.image_dim = image_dim

        self.manager_image_encoder = ImageEncoder(
            obs_space,
            image_dim=image_dim,
            arch=arch
        )
        self.actor_image_encoder = ImageEncoder(
            obs_space,
            with_manager_map=True,
            image_dim=image_dim,
            arch=arch
        )

        self.manager_action_latent_embedding = nn.Embedding(action_latent_size, 32)
        self.manager_observation_latent_embeddings = [
            nn.Embedding(observation_latent_size, 32)
            for i in range(observation_latents)
        ]
        self.manager_observation_latent_code = nn.Linear(
            32 * observation_latents,
            128
        )

        # Define instruction embedding
        if self.use_instr:
            self.language_encoder = LanguageEncoder(
                obs_space,
                instr_dim,
                lang_model=lang_model,
                arch=arch
            )

            self.memory_language_attention = MemoryLanguageAttention(
                instr_dim,
                memory_dim=memory_dim,
                lang_model=lang_model,
                arch=arch
            )

            self.film_image_conditioning = FiLMImageConditioning(
                self.image_dim,
                self.language_encoder.final_instr_dim,
                arch=arch
            )

        self.actor_film_image_conditioning = FiLMImageConditioning(
            self.image_dim,
            32,
            arch=arch
        )
        self.manager_actor_image_conv1 = nn.Conv2d(128, 1, kernel_size=(1, 1))
        self.film_pooling = FiLMPooling(arch=arch)

        if self.use_memory:
            self.memory = Memory(
                64,
                memory_dim=memory_dim
            )
            self.actor_memory = Memory(
                image_dim,
                memory_dim=memory_dim
            )
            self.embedding_size = self.memory.semi_memory_size
        else:
            self.embedding_size = image_dim

        self.manager_heatmap = Heatmap(
            in_channels=self.image_dim,
            embedding_size=observation_latent_size
        )

        # Define memory and resize image embedding
        self.embedding_size = image_dim
        self.action_latent_size = action_latent_size
        self.observation_latent_size = observation_latent_size
        self.n_latent_observations = observation_latents
        self.action_space = action_space
        self.countdown = 0

        self.manager_observations = ManagerObservations(
            self.embedding_size,
            self.language_encoder.final_instr_dim,
            self.observation_latent_size,
            observation_latents
        )

        self.manager_observations_lang_embedding_reconstr = nn.Sequential(
            nn.Linear(observation_latents * 32, instr_dim),
            nn.Tanh(),
            nn.Linear(instr_dim, instr_dim)
        )

        # Define the manager model. The manager sees everything.
        self.manager = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_latent_size)
        )

        # Define actor's model. The actor sees only the image
        # latent before FiLM has been applied.
        self.actor = nn.Sequential(
            nn.Linear(self.image_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model. The critic sees everything,
        # since it needs to estimate the value function.
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        self.extra_heads = ExtraHeads(self.embedding_size, aux_info)

    def add_heads(self):
        self.extra_heads.add_heads()

    def add_extra_heads_if_necessary(self, aux_info):
        self.extra_heads.add_extra_heads_if_necessary(self, aux_info)

    @property
    def memory_size(self):
        return self.memory.memory_size

    @property
    def semi_memory_size(self):
        return self.memory.semi_memory_size

    def forward(self, obs, actor_memory, manager_memory, instr_embedding=None, manager_action_latent=None):
        # manager_observation_latents = self.manager_observation_latent_code(
        #     torch.cat([
        #         torch.mean(e.weight, dim=0)
        #         for e in self.manager_observation_latent_embeddings
        #     ], dim=-1)
        #     if manager_observation_latent is None
        #     else torch.cat([
        #         embedding, latent in
        #         zip(
        #             self.manager_observation_latent_embeddings,
        #             manager_observation_latents.permute(1, 0)
        #         )
        #     ], dim=-1)
        # )
        manager_action_mask = make_observation_masks(
            manager_action_latent,
            self.action_latent_size,
            obs.direction,
            obs.image.shape[0],
            obs.image.shape[1],
            obs.image.shape[2],
            device=obs.image.device
        )

        # Rotate manager input images to face in a front-facing direction
        forward_facing_images = rotate_image_batch(obs.image, obs.direction)
        manager_img_encoding = self.manager_image_encoder(forward_facing_images)
        actor_img_encoding = self.actor_image_encoder(
            obs.image,
            manager_action_mask
        )

        # Produce encodings
        if self.use_instr and instr_embedding is None:
            instr_embedding = self.language_encoder(obs.instr)

        if self.use_instr:
            instr_embedding = self.memory_language_attention(
                obs.instr,
                instr_embedding,
                manager_memory
            )

            manager_attended_img = self.film_image_conditioning(manager_img_encoding, instr_embedding if self.use_instr else None)
        else:
            manager_attended_img = manager_img_encoding

        # Manager generates a heatmap from the image based on what it knows about
        # the problem. Eg, "go to the blue box" should highlight the blue box.
        #manager_map = self.manager_heatmap(img_encoding, manager_observation_latents)

        # Now generate the information required for the manager
        manager_pooled_attended_img = self.manager_actor_image_conv1(manager_attended_img).reshape(-1, 64)

        # UNUSED Inductive bias: The manager map is a form of attention on the original image -
        # before max-pooling, the manager map highlights what we think is going to be
        # important for the actor to be able to do its job.
        actor_pooled_img = self.film_pooling(actor_img_encoding)

        if self.use_memory:
            embedding, manager_memory = self.memory(manager_pooled_attended_img, manager_memory)
            embedding_actor, actor_memory = self.actor_memory(actor_pooled_img, actor_memory)
        else:
            embedding = manager_pooled_attended_img
            embedding_actor = actor_pooled_img

        x = self.actor(embedding_actor)
        dist = Categorical(logits=x)

        # manager_observations = self.manager_observations(embedding, instr_embedding)
        # manager_observations_dists = Categorical(logits=manager_observations)

        x = self.manager(embedding)
        manager_dist = Categorical(logits=x)

        x = self.critic(embedding_actor)
        value = x.squeeze(1)

        extra_predictions = self.extra_heads(embedding)

        # Take the argmax from the manager observation distributions
        # and try to reconstruct the language observation, returning
        # it as a loss. Then multiply with the probability of the
        # language instructions and take the sum
        # manager_observations_argmax = manager_observation_dists.argmax(dim=-1)
        # manager_observations_embedding = torch.cat([
        #     layer(argmax_batch)
        #     for layer, argmax_batch in zip(
        #         self.manager_observation_latent_embeddings,
        #         manager_observations_argmax.permute(1, 0)
        #     )
        # ], dim=-1)
        # lang_reconstruction = self.manager_observations_lang_embedding_reconstr(manager_observations_embedding)
        # manager_lang_reconstruction_loss = torch.nn.functional.mse_loss(lang_reconstruction, instr_embedding, reduction='none').mean(dim=-1)
        # manager_observations_log_probs = manager_observation_dists.log_prob(manager_observations_argmax)

        # We take the sum here, which is the same as the total probability
        # of seeing this reconstruction loss, then exponentiate. Then we can
        # 
        # prob_weighted_reconstruction_loss = manager_lang_reconstruction_loss * torch.exp(manager_observations_log_probs.sum(dim=-1)).mean()

        return {
            'dist': dist,
            'manager_dist': manager_dist,
            # 'manager_observations_dists': manager_observations_dists,
            # 'manager_reconstruction_loss': prob_weighted_reconstruction_loss,
            # 'lang_embedding': lang_embedding,
            'value': value,
            'memory': actor_memory,
            'manager_memory': manager_memory,
            'extra_predictions': extra_predictions
        }