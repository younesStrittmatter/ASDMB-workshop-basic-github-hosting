import psyneulink as pnl
import numpy as np
import pandas as pd
import random

from psyneulink.core.globals.utilities import set_global_seed

set_global_seed(0)

color_map = lambda x: "red" if x == [1, 0] else "green"
word_map = lambda x: "RED" if x == [1, 0] else "GREEN"
task_map = lambda x: "word reading" if x == [1, 0] else "color naming"
decision_map = lambda x: "red" if x == -1 else "green"


class Stroop:
    def __init__(self):
        """
        :param word_w: Weight of the word projections
        :param color_w: Weight of the color projections
        :param task_w: Weight of the task projections
        """

        self.comp = pnl.Composition(name='Stroop')

    def predict(self, x):
        self.comp.run(x, execution_mode=pnl.ExecutionMode.LLVMRun)
        return self.comp.results

    def fit(self, x, y):
        import optuna
        fit_parameters = {
            ("threshold", self.decision): np.linspace(0.01, 40, 1000),  # threshold
            ("rate", self.decision): np.linspace(0.0, 0.1, 1000),  # rate
        }

        pec = pnl.ParameterEstimationComposition(
            name="pec",
            nodes=self.comp,
            parameters=fit_parameters,
            outcome_variables=[
                self.decision.output_ports[0],
                self.decision.output_ports[1],
            ],
            data=y,
            optimization_function=pnl.PECOptimizationFunction(method=optuna.samplers.CmaEsSampler(seed=0),
                                                              max_iterations=1000),
            num_estimates=5000,
        )

        pec.controller.parameters.comp_execution_mode.set("LLVM")
        pec.controller.function.parameters.save_values.set(True)

        ret = pec.run(inputs=x)
        optimal_parameters = list(pec.optimized_parameter_values.values())
        print(optimal_parameters)

    def report(self):
        print('Nothing to report yet...')
        # print('** INPUTS **')
        # print(f'Word: {self.word.value} ({word_map(list(self.word.value[0]))})')
        # print(f'Color: {self.color.value} ({color_map(list(self.color.value[0]))})')
        # print(f'Task: {self.task.value} ({task_map(list(self.task.value[0]))})')
        #
        # print('** Hidden **')
        # print(f'Word (hidden): {self.word_hidden.value}')
        # print(f'Color (hidden): {self.color_hidden.value}')
        # print(f'Task (hidden): {self.task_hidden.value}')
        #
        # print('** OUTPUTS **')
        # print(
        #     f'Output: {self.output.value} ({"red > green" if self.output.value[0][0] > self.output.value[0][1] else "green > red"})')
        #
        # print('** Conflict **')
        # print(f'conflict: {self.conflict.value}')
        #
        # print('** Decision **')
        # print(self.decision.output_ports[0].value, self.decision.output_ports[1].value)
        # print('*' * 20)
        # print()

    def generate_random_inputs(self, n):
        import numpy as np
        # choices = [[1, 0], [0, 1]]
        # return {
        #     self.word: [choices[i] for i in np.random.randint(0, 2, size=n)],
        #     self.color: [choices[i] for i in np.random.randint(0, 2, size=n)],
        #     self.task: [choices[i] for i in np.random.randint(0, 2, size=n)],
        # }

    def show(self):
        self.comp.show_graph()

    def test_run(self):
        random_inputs = self.generate_random_inputs(5)
        self.comp.run(random_inputs, execution_mode=pnl.ExecutionMode.LLVMRun, callback=self.report)

    def test_fit(self):
        random_inputs = self.generate_random_inputs(2)
        self.comp.run(random_inputs, execution_mode=pnl.ExecutionMode.LLVMRun)

        data_to_fit = pd.DataFrame(
            np.squeeze(np.array(self.comp.results))[:, 1:], columns=["decision", "response_time"]
        )

        data_to_fit["decision"] = data_to_fit["decision"].astype("category")

        self.fit(random_inputs, data_to_fit)


if __name__ == '__main__':
    my_stroop = Stroop()
    my_stroop.show()

    # my_stroop.test_run()
    # my_stroop.test_fit()
