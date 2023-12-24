from cfl.src.read_data import *
from cfl.src.agent_model import AgentModel
from cfl.src.federated_averaging import FederatedAveraging, IsolatedTraining
from cfl.tools.performance_evaluator import ClassificationEvaluator as MyClassifierEvaluator
from cfl.tools.convergence_evaluator import NumberOfRounds

from sklearn.preprocessing import StandardScaler as MyDataNormalizer
from sklearn.model_selection import train_test_split
from copy import deepcopy

EPSILON = 1e-20


class FedAvePipeLine(object):
    def __init__(self, agent_ids: list,
                 data_obj,
                 test_ratio: float = 0.9,
                 val_ratio: {float, None}=0.1,
                 ):
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.agent_ids = agent_ids
        self.sr = data_obj

        self.data_dict = {}
        for target_agent in agent_ids:
            data_dict = self.sr.read(agent_id=target_agent)
            if isinstance(data_dict['dataset']['X'], pd.DataFrame):
                x = data_dict['dataset']['X'].to_numpy()
                y = data_dict['dataset']['Y'][self.sr.tasks].to_numpy()
            else:
                x = data_dict['dataset']['X']
                y = data_dict['dataset']['Y'].reshape(-1, 1)
            x_train_target, x_test_target, y_train_target, y_test_target = \
                train_test_split(x, y, test_size=self.test_ratio, shuffle=False)

            self.data_dict.update({target_agent: {'x_train': x_train_target,
                                                  'x_test': x_test_target,
                                                  'y_train': y_train_target,
                                                  'y_test': y_test_target}
                                   })


class AgentLocalTraining(FedAvePipeLine):
    def __init__(self, local_trainer,
                 target_agent: str,
                 agent_ids: list,
                 data_obj,
                 test_ratio: float,
                 val_ratio: (float, None)):
        super().__init__(agent_ids,
                         data_obj,
                         test_ratio,
                         val_ratio)
        self.target_agent = target_agent
        self.am = AgentModel(federated_scheme=IsolatedTraining(),
                             data_normalizer=MyDataNormalizer(),
                             local_trainer=deepcopy(local_trainer),
                             evaluator=MyClassifierEvaluator(tasks=self.sr.tasks),
                             validation_ratio=self.val_ratio)
        self._target_agent_local_trainer = None
        self._eval_dict = None

    @property
    def target_agent_local_trainer(self):
        return self._target_agent_local_trainer

    @property
    def eval_dict(self):
        return self._eval_dict

    def fit_predict_evaluate(self):
        target_data = self.data_dict[self.target_agent]
        self.am.fit_predict_evaluate(x_train=target_data['x_train'],
                                     y_train=target_data['y_train'],
                                     x_test=target_data['x_test'],
                                     y_test=target_data['y_test']
                                     )
        print('local training agent %s: %s' % (self.target_agent, self.am.eval_dict))
        self._eval_dict = self.am.eval_dict
        self._target_agent_local_trainer = self.am.local_trainer


class AgentMultiRoundProbConditionalFederatedAveraging(FedAvePipeLine):
    def __init__(self, local_trainer,
                 n_rounds: int,
                 agent_ids: list,
                 data_obj,
                 test_ratio: float,
                 val_ratio: {None, float},
                 num_epochs: int,
                 use_best_model: bool,
                 std_thr: {float, str},
                 print_key: str,
                 if_print: bool = True):
        super().__init__(agent_ids,
                         data_obj,
                         test_ratio,
                         val_ratio)
        self.print_key = print_key
        self.std_thr = std_thr
        self.use_best_model = use_best_model
        self.num_epochs = num_epochs
        self._global_trainer = deepcopy(local_trainer)
        self._federated_trainer = None
        self._federated_trainer_std = None
        self._federated_trainer_sample = None
        self._eval_dict_list = []
        self.fa = FederatedAveraging(convergence_evaluator=NumberOfRounds(n_rounds=n_rounds))
        self.if_print = if_print

    @property
    def eval_dict_list(self):
        return self._eval_dict_list

    @property
    def federated_trainer(self):
        return self._federated_trainer

    @property
    def federated_trainer_std(self):
        return self._federated_trainer_std

    @property
    def federated_trainer_sample(self):
        return self._federated_trainer_sample

    def fit_predict_evaluate_init(self, mu=None):
        federated_trainer_std = None
        federated_trainer = None
        federated_trainer_sample = None
        agent_model_dict = {}
        r = 0
        while self.fa.convergence_evaluator.terminate(r) is False:
            if self.if_print is True:
                print('round %s' % str(r))
            eval_dict = {}
            if r == 0:
                for name in self.agent_ids:
                    am = AgentModel(federated_scheme=self.fa,
                                    data_normalizer=MyDataNormalizer(),
                                    local_trainer=deepcopy(self._global_trainer[name]),
                                    evaluator=MyClassifierEvaluator(self.sr.tasks),
                                    validation_ratio=self.val_ratio)
                    am.fit_predict_evaluate(x_train=self.data_dict[name]['x_train'],
                                            y_train=self.data_dict[name]['y_train'],
                                            x_test=self.data_dict[name]['x_test'],
                                            y_test=self.data_dict[name]['y_test'])
                    agent_model_dict.update({name: am.local_trainer})
                    eval_dict.update({name: am.eval_dict})
                federated_trainer = self.fa.average(agent_model_dict=agent_model_dict)
                federated_trainer_std = self.fa.std(agent_model_dict=agent_model_dict, std_thr=self.std_thr)
                federated_trainer_sample = self.fa.sample(average_model=federated_trainer,
                                                          std_model=federated_trainer_std)
            else:
                for name in self.agent_ids:
                    sample_model = self.fa.sample(federated_trainer, federated_trainer_std)

                    am = self.update_agent_model(aggregated_trainer=deepcopy(sample_model),
                                                 reference_trainer=deepcopy(federated_trainer),
                                                 reference_trainer_std=deepcopy(federated_trainer_std),
                                                 num_epochs=self.num_epochs)
                    am.local_trainer.set('_prox_term', mu)
                    am.fit_predict_evaluate(x_train=self.data_dict[name]['x_train'],
                                            y_train=self.data_dict[name]['y_train'],
                                            x_test=self.data_dict[name]['x_test'],
                                            y_test=self.data_dict[name]['y_test'])
                    agent_model_dict.update({name: am.local_trainer})
                    eval_dict.update({name: am.eval_dict})
                    if self.if_print is True:
                        print(self.print_key + ' %s: %s' % (name, am.eval_dict))

                federated_trainer = self.fa.average(agent_model_dict=agent_model_dict)
                federated_trainer_std = self.fa.std(agent_model_dict=agent_model_dict)
                federated_trainer_sample = self.fa.sample(average_model=federated_trainer,
                                                          std_model=federated_trainer_std)

            self._eval_dict_list.append(eval_dict)
            r += 1
        self._federated_trainer = deepcopy(federated_trainer)
        self._federated_trainer_std = deepcopy(federated_trainer_std)
        self._federated_trainer_sample = deepcopy(federated_trainer_sample)

    def fit_predict_evaluate(self, mu=None):
        federated_trainer = self.fa.average(agent_model_dict=deepcopy(self._global_trainer))
        federated_trainer_std = self.fa.std(agent_model_dict=deepcopy(self._global_trainer))
        federated_trainer_sample = None
        agent_model_dict = {}
        r = 0
        while self.fa.convergence_evaluator.terminate(r) is False:
            if self.if_print is True:
                print('round %s' % str(r))
            eval_dict = {}
            for name in self.agent_ids:
                sample_model = self.fa.sample(federated_trainer, federated_trainer_std)

                am = self.update_agent_model(aggregated_trainer=deepcopy(sample_model),
                                             reference_trainer=deepcopy(federated_trainer),
                                             reference_trainer_std=deepcopy(federated_trainer_std),
                                             num_epochs=self.num_epochs)
                am.local_trainer.set('_prox_term', mu)
                am.fit_predict_evaluate(x_train=self.data_dict[name]['x_train'],
                                        y_train=self.data_dict[name]['y_train'],
                                        x_test=self.data_dict[name]['x_test'],
                                        y_test=self.data_dict[name]['y_test'])
                agent_model_dict.update({name: am.local_trainer})
                eval_dict.update({name: am.eval_dict})
                if self.if_print is True:
                    print(self.print_key + ' %s: %s' % (name, am.eval_dict))

            federated_trainer = self.fa.average(agent_model_dict=agent_model_dict)
            federated_trainer_std = self.fa.std(agent_model_dict=agent_model_dict)
            federated_trainer_sample = self.fa.sample(average_model=federated_trainer,
                                                      std_model=federated_trainer_std)

            self._eval_dict_list.append(eval_dict)
            r += 1
        self._federated_trainer = deepcopy(federated_trainer)
        self._federated_trainer_std = deepcopy(federated_trainer_std)
        self._federated_trainer_sample = deepcopy(federated_trainer_sample)

    def update_agent_model(self, aggregated_trainer, reference_trainer, reference_trainer_std, num_epochs):
        aggregated_local_trainer = deepcopy(aggregated_trainer)

        aggregated_local_trainer.num_epochs = num_epochs
        layers = list(aggregated_local_trainer.model.state_dict().keys())
        weight_layers = list(filter(lambda j_layer: 'weight' in j_layer, layers))

        mask_names = list(reference_trainer.model.mask_dict.keys())
        std_mask = list(filter(lambda m: 'std' in m, mask_names))
        for _ in std_mask:
            mask_names.remove(_)
        for i, mask_name in zip(range(len(mask_names)), mask_names):
            layer_mask = reference_trainer.model.state_dict()[weight_layers[i]]
            layer_mask_std = reference_trainer_std.model.state_dict()[weight_layers[i]]
            aggregated_local_trainer.model.mask_dict.update({mask_name: layer_mask})
            aggregated_local_trainer.model.mask_dict.update({mask_name + '_std': layer_mask_std})

        aggregated_local_trainer.use_best_model = self.use_best_model
        am = AgentModel(federated_scheme=IsolatedTraining(),
                        data_normalizer=MyDataNormalizer(),
                        local_trainer=deepcopy(aggregated_local_trainer),
                        evaluator=MyClassifierEvaluator(tasks=self.sr.tasks),
                        validation_ratio=self.val_ratio)
        return am


class AgentMultiRoundAveraging(FedAvePipeLine):
    def __init__(self, local_trainer,
                 agent_ids: list,
                 data_obj,
                 test_ratio: float,
                 val_ratio: {None, float},
                 std_thr: float,
                 use_best_model: bool,
                 print_key: str,
                 if_print: bool = True):
        super().__init__(agent_ids,
                         data_obj,
                         test_ratio,
                         val_ratio)
        self.print_key = print_key
        self.std_thr = std_thr
        self._global_trainer = deepcopy(local_trainer)
        self.use_best_model = use_best_model
        self._federated_trainer = None
        self._federated_trainer_std = None
        self._eval_dict = None
        self.fa = FederatedAveraging(convergence_evaluator=NumberOfRounds(n_rounds=1))
        self.if_print = if_print

    @property
    def eval_dict(self):
        return self._eval_dict

    @property
    def federated_trainer(self):
        return self._federated_trainer

    @property
    def federated_trainer_std(self):
        return self._federated_trainer_std

    def average_predict_evaluate(self, agent_model_dict):
        eval_dict = {}
        federated_trainer = self.fa.average(agent_model_dict=agent_model_dict)
        federated_trainer_std = self.fa.std(agent_model_dict=agent_model_dict, std_thr=self.std_thr)
        for name in self.agent_ids:
            am = AgentModel(federated_scheme=IsolatedTraining(),
                            data_normalizer=MyDataNormalizer(),
                            local_trainer=deepcopy(federated_trainer),
                            evaluator=MyClassifierEvaluator(tasks=self.sr.tasks),
                            validation_ratio=self.val_ratio)
            am.predict_evaluate(x_train=self.data_dict[name]['x_train'],
                                y_train=self.data_dict[name]['y_train'],
                                x_test=self.data_dict[name]['x_test'],
                                y_test=self.data_dict[name]['y_test'])
            eval_dict.update({name: am.eval_dict})
            if self.if_print is True:
                print(self.print_key + ' %s: %s' % (name, am.eval_dict))

        self._eval_dict = eval_dict
        self._federated_trainer = deepcopy(federated_trainer)
        self._federated_trainer_std = deepcopy(federated_trainer_std)


class AgentMultiRoundFederatedAveraging(FedAvePipeLine):
    def __init__(self, local_trainer,
                 n_rounds: int,
                 agent_ids: list,
                 data_obj: SimpleRead,
                 test_ratio: {float, None},
                 val_ratio: {None, float},
                 if_print: bool = True):
        super().__init__(agent_ids,
                         data_obj,
                         test_ratio,
                         val_ratio)
        self.if_print = if_print
        self._global_trainer = deepcopy(local_trainer)
        self._federated_trainer = None
        self._federated_trainer_std = None
        self._eval_dict_list = []
        self.fa = FederatedAveraging(convergence_evaluator=NumberOfRounds(n_rounds=n_rounds))

    @property
    def eval_dict_list(self):
        return self._eval_dict_list

    @property
    def federated_trainer(self):
        return self._federated_trainer

    @property
    def federated_trainer_std(self):
        return self._federated_trainer_std

    def fit_predict_evaluate(self):
        federated_trainer = deepcopy(self._global_trainer)
        agent_model_dict = {}
        r = 0
        while self.fa.convergence_evaluator.terminate(r) is False:
            if self.if_print is True:
                print('round %s' % str(r))
            eval_dict = {}
            for name in self.agent_ids:
                am = AgentModel(federated_scheme=self.fa,
                                data_normalizer=MyDataNormalizer(),
                                local_trainer=deepcopy(federated_trainer),
                                evaluator=MyClassifierEvaluator(self.sr.tasks),
                                validation_ratio=self.val_ratio)
                am.fit_predict_evaluate(x_train=self.data_dict[name]['x_train'],
                                        y_train=self.data_dict[name]['y_train'],
                                        x_test=self.data_dict[name]['x_test'],
                                        y_test=self.data_dict[name]['y_test'])
                if self.if_print is True:
                    print('federated training agent %s: %s' % (name, am.eval_dict))
                agent_model_dict.update({name: am.local_trainer})
                eval_dict.update({name: am.eval_dict})
            self._eval_dict_list.append(eval_dict)
            federated_trainer = self.fa.average(agent_model_dict=agent_model_dict)
            r += 1
            if r == self.fa.convergence_evaluator.n_rounds:
                self._federated_trainer_std = self.fa.std(agent_model_dict=agent_model_dict)
        self._federated_trainer = deepcopy(federated_trainer)
