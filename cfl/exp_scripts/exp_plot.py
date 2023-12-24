import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib


class ExpPlot(object):
    def __init__(self, save_path, 
                n_agents, 
                task_list,
                n_rounds,
                n_runs, 
                epoch_list, 
                make_predictions_using_local_model):
        self.save_path = save_path
        self.n_runs = n_runs
        self.n_agents = n_agents
        self.task_list = task_list
        self.n_rounds = n_rounds
        self.epoch_list = epoch_list
        self.make_predictions_using_local_model = make_predictions_using_local_model

    def __call__(self):
        poi_list = list(np.linspace(0, self.n_rounds - 1, min(10, self.n_rounds), dtype='int'))
        agent_names = [str(_) for _ in range(self.n_agents)]
        main_path = self.save_path
        fig_path = main_path + 'figs/'
        methods = []
        method_names = {}
        colors = []
        alphas = []
        hatch_list = []
        hatches = ['..', '.', 'o', 'O']
        for epoch, j in zip(self.epoch_list, range(len(self.epoch_list))):
            methods += [f'fl{epoch}', f'cfl{epoch}']
            method_names.update({f'fl{epoch}': f'FL-{epoch}', f'cfl{epoch}': f'CFL-{epoch}'})
            colors += ['#8f0626', '#062d8f']
            alphas += [1. - j/len(self.epoch_list), 1. - j/len(self.epoch_list)]
            hatch_list += [hatches[j], hatches[j]]

        for task_i in self.task_list:
            this_tasks = [task_i]

            color_dict = {}
            alpha_dict = {}
            hatch_dict = {}
            for j, method in zip(range(len(methods)), methods):
                color_dict.update({method: colors[j]})
                alpha_dict.update({method: alphas[j]})
                hatch_dict.update({method: hatch_list[j]})
            matplotlib.rcParams.update({'font.size': 16})
            fig, ax = plt.subplots(figsize=(15, 5))
            x_pos = np.arange(len(poi_list))
            metric = 'f1-score'
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            kpi_method = {}
            for method in methods:
                if method.__contains__('cfl') and self.make_predictions_using_local_model is True:
                    file_name = method + '_local__kpi'
                else:
                    file_name = method + '__kpi'
                kpi_agent = {}
                for agent_name in agent_names:
                    kpi_task = {}
                    for task in this_tasks:
                        kpi_list = []
                        for exp_id in range(self.n_runs):
                            file_path = main_path + 'agent_' + agent_name + '/' + str(exp_id) + '/'
                            result_path = file_path + file_name
                            res_dict = pd.read_pickle(result_path + '.pickle')
                            kpi_list.append([res_dict[i][task][metric] for i in range(self.n_rounds)])
                            kpi_task.update({task: np.atleast_2d(kpi_list).T})
                    kpi_agent.update({agent_name: kpi_task})
                kpi_method.update({method: kpi_agent})

            min_kpi = np.inf
            max_kpi = -np.inf
            for p_id, POI in zip(range(len(poi_list)), poi_list):
                kpi_all_dict = {}
                for method in methods:
                    mean_all = np.mean(
                        [np.mean(kpi_method[method][agent_name][this_tasks[0]][POI:]) for agent_name in agent_names])
                    std_all = np.mean(
                        [np.std(kpi_method[method][agent_name][this_tasks[0]][POI:]) for agent_name in agent_names])
                    kpi_all_dict.update({method: [mean_all, std_all]})
                    min_kpi = min(mean_all, min_kpi)
                    max_kpi = max(mean_all, max_kpi)

                # -----------------------------------------------------------------------------
                for i, method in zip(range(len(methods)), methods):
                    if p_id == 0:
                        lab = method_names[method]
                    else:
                        lab = None

                    ax.bar(x_pos[p_id] + (i * .12), kpi_all_dict[method][0], color=color_dict[method],
                           label=lab,
                           yerr=kpi_all_dict[method][1],
                           align='center', alpha=alpha_dict[method], ecolor='black',
                           error_kw=dict(lw=.5, capsize=.5, capthick=.5),
                           width=.11, fill=True,  hatch=hatch_dict[method])
                    ax.bar(x_pos[p_id] + (i * .12), kpi_all_dict[method][0], color=color_dict[method],
                           label=None,
                           yerr=kpi_all_dict[method][1],
                           align='center', alpha=alpha_dict[method], ecolor='black',
                           error_kw=dict(lw=.5, capsize=.5, capthick=.5),
                           width=.1, fill=False,  hatch=hatch_dict[method])
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=8, fancybox=False, shadow=False,
                          prop={'size': 13})
            plt.xticks(range(len(poi_list)), poi_list)
            plt.ylabel(metric)
            plt.ylim([min_kpi*.95, max_kpi*1.05])
            plt.xlabel('Round')
            plt.grid(True, color='y', linestyle='-', linewidth=.4)
            if not os.path.exists(fig_path + task_i + '/'):
                os.makedirs(fig_path + task_i + '/')
            plt.tight_layout()
            plt.savefig(fig_path + task_i + '/poi_' + metric + '.jpg', dpi=1200)
            plt.show()
