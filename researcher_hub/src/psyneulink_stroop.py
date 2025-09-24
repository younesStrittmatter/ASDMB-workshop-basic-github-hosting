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
    def __init__(self, word_w=3, color_w=2, task_w=4):
        """
        :param word_w: Weight of the word projections
        :param color_w: Weight of the color projections
        :param task_w: Weight of the task projections
        """

        self.comp = pnl.Composition(name='Stroop')

        self.word = pnl.ProcessingMechanism(name="Word", input_shapes=2)
        self.color = pnl.ProcessingMechanism(name="Color", input_shapes=2)
        self.task = pnl.ProcessingMechanism(name="Task", input_shapes=2, function=pnl.Linear)

        self.word_hidden = pnl.ProcessingMechanism(name="Word (Hidden)", input_shapes=2, function=pnl.Logistic(bias=-4))
        self.color_hidden = pnl.ProcessingMechanism(name="Color (Hidden)", input_shapes=2,
                                                    function=pnl.Logistic(bias=-4))
        self.task_hidden = pnl.ProcessingMechanism(name="Task (Hidden)", input_shapes=2,
                                                   function=pnl.Logistic)

        self.output = pnl.ProcessingMechanism(name="Output", input_shapes=2)

        self.comp.add_linear_processing_pathway([
            self.word,
            [[word_w, -word_w], [-word_w, word_w]],
            self.word_hidden
        ])
        self.comp.add_linear_processing_pathway([
            self.color,
            [[color_w, -color_w], [-color_w, color_w]],
            self.color_hidden
        ])

        self.comp.add_linear_processing_pathway([
            self.task,
            self.task_hidden
        ])

        self.comp.add_linear_processing_pathway([
            self.task_hidden,
            [[task_w, task_w], [0, 0]],
            self.word_hidden
        ])

        self.comp.add_linear_processing_pathway([
            self.task_hidden,
            [[0, 0], [task_w, task_w]],
            self.color_hidden
        ])

        self.comp.add_linear_processing_pathway([
            self.word_hidden,
            self.output
        ])

        self.comp.add_linear_processing_pathway([
            self.color_hidden,
            self.output
        ])

        ### Add Conflict Monitoring
        self.conflict = pnl.ProcessingMechanism(
            name='Conflict',
            input_shapes=1,
            function=lambda x: 1 / (x * x + 1),
        )

        self.comp.add_linear_processing_pathway([
            self.output,
            [[1], [-1]],
            self.conflict
        ])

        self.control = pnl.ControlMechanism(
            name='Control',
            objective_mechanism=pnl.ObjectiveMechanism(
                name='Conflict Monitor',
                monitor=self.conflict,
            ),
            default_allocation=[.5],
            control_signals=[(pnl.GAIN, self.task_hidden)]
        )

        self.comp.add_controller(controller=self.control)

        self.decision = pnl.DDM(
            name='Decision',
            input_format=pnl.ARRAY,
            function=pnl.DriftDiffusionIntegrator(
                noise=.5,
                rate=.05,
                threshold=20.,
            ),
            output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
            reset_stateful_function_when=pnl.AtTrialStart())

        self.out_vars = [
            self.decision.output_ports[0],
            self.decision.output_ports[1],
        ]

        self.comp.add_linear_processing_pathway([
            self.output,
            self.decision,
        ])

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


        # print('Fitting model...')
        # y = y.copy()
        #
        # # Parameter grid (keys are (parameter_name, owner) tuples)
        # params = {
        #     ("threshold", self.comp.nodes['Decision']): np.linspace(0.01, 0.5, 1000),  # Threshold
        # }
        #
        # # Optional conditioning (e.g., different thresholds per condition)
        # # depends = None
        # # if condition_col is not None:
        # #     if condition_col not in y:
        # #         raise ValueError(f"condition_col '{condition_col}' not found in y")
        # #     depends = {('non_decision_time', self.comp.nodes['Decision']): 'condition'}
        # import optuna
        # pec = pnl.ParameterEstimationComposition(
        #     model=self.comp,
        #     parameters=params,
        #     outcome_variables=self.out_vars,
        #     # must be terminal ports
        #     data=y,
        #     optimization_function=pnl.PECOptimizationFunction(method=optuna.samplers.CmaEsSampler(seed=0),
        #                                                       max_iterations=1000),
        #     initial_seed=42,
        #     same_seed_for_all_parameter_combinations=True,
        #     num_estimates=100
        # )
        # pec.controller.parameters.comp_execution_mode.set("LLVM")
        # # pec.controller.function.parameters.save_values.set(True)
        #
        # pec.run(inputs=self._inputs_from_X(x))
        # return pec.optimized_parameter_values

    def report(self):
        print('** INPUTS **')
        print(f'Word: {self.word.value} ({word_map(list(self.word.value[0]))})')
        print(f'Color: {self.color.value} ({color_map(list(self.color.value[0]))})')
        print(f'Task: {self.task.value} ({task_map(list(self.task.value[0]))})')

        print('** Hidden **')
        print(f'Word (hidden): {self.word_hidden.value}')
        print(f'Color (hidden): {self.color_hidden.value}')
        print(f'Task (hidden): {self.task_hidden.value}')

        print('** OUTPUTS **')
        print(
            f'Output: {self.output.value} ({"red > green" if self.output.value[0][0] > self.output.value[0][1] else "green > red"})')

        print('** Conflict **')
        print(f'conflict: {self.conflict.value}')

        print('** Decision **')
        print(self.decision.output_ports[0].value, self.decision.output_ports[1].value)
        # print(f'output_layer: {self.output_layer.value}')
        # print(f'conflict: {self.conflict.value}')
        # print(f'task demand: {self.task_demand_input.value}')
        # print(f'task hidden: {self.task_hidden.value}')
        # print()
        print('*' * 20)
        print()

    def generate_random_inputs(self, n):
        import numpy as np
        choices = [[1, 0], [0, 1]]
        return {
            self.word: [choices[i] for i in np.random.randint(0, 2, size=n)],
            self.color: [choices[i] for i in np.random.randint(0, 2, size=n)],
            self.task: [choices[i] for i in np.random.randint(0, 2, size=n)],
        }

    def generate_random_outputs(self, n):
        return {
            pnl.DECISION_OUTCOME: random.choices([0, 1], k=n),
            pnl.RESPONSE_TIME: [random.random() * 2000 for _ in range(n)]
        }

    def show(self):
        self.comp.show_graph()

    def test(self):
        random_inputs = self.generate_random_inputs(2)
        self.comp.run(random_inputs, execution_mode=pnl.ExecutionMode.LLVMRun)


        data_to_fit = pd.DataFrame(
            np.squeeze(np.array(self.comp.results))[:, 1:], columns=["decision", "response_time"]
        )

        data_to_fit["decision"] = data_to_fit["decision"].astype("category")


        self.fit(random_inputs, data_to_fit)

        #     print()


if __name__ == '__main__':
    my_stroop = Stroop()
    my_stroop.test()

    # my_stroop.show()
