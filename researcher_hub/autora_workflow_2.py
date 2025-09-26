import json
import random

import psyneulink as pnl

from autora.variable import VariableCollection, Variable
from autora.experimentalist.random import pool
from autora.experiment_runner.firebase_prolific import firebase_runner
from autora.state import StandardState, on_state, Delta
from src.psyneulink_stroop import Stroop

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sweetpea import Factor, DerivedLevel, CrossBlock, synthesize_trials, experiments_to_dicts, WithinTrial

from sweetbean.stimulus import Fixation, ROK, Feedback, Text
from sweetbean.variable import FunctionVariable, TimelineVariable
from sweetbean import Block, Experiment
from sweetbean.extension import TouchButton
from sweetbean.util.data_process import process_autora

# *** Set up variables *** #
variables = VariableCollection(
    independent_variables=[
        Variable(name="trial_sequence", allowed_values=[]),
    ],
    dependent_variables=[
        Variable(name="responses")])

# *** State *** #

state = StandardState(
    variables=variables,
)

# *** Components/Agents *** #
# Components are functions that run on the state. The main components are:
# - theorist
# - experiment-runner
# - experimentalist
# See more about components here: https://autoresearch.github.io/autora/


# ** Theorist ** #
# Here we use a linear regression as theorist, but you can use other theorists included in
# autora (for a list: https://autoresearch.github.io/autora/theorist/)

# theorist = LinearRegression()
theorist = Stroop()


# To use the theorist on the state object, we wrap it with the on_state functionality and return a
# Delta object.
# Note: The if the input arguments of the theorist_on_state function are state-fields like
# experiment_data, variables, ... , then using this function on a state object will automatically
# use those state fields.
# The output of these functions is always a Delta object. The keyword argument in this case, tells
# the state object witch field to update.


@on_state()
def theorist_on_state(experiment_data):
    theorist = Stroop()
    all_conditions = {
        theorist.color: [],
        theorist.word: [],
        theorist.task: [],
    }
    all_observations = {
        'decision': [],
        'reaction_time': [],
    }

    for _, e in experiment_data.iterrows():
        conditions = e['congruency_freq']
        observations = e['congruency_effect']
        all_conditions[theorist.color].extend(conditions['color'])
        all_conditions[theorist.word].extend(conditions['word'])
        all_conditions[theorist.task].extend(conditions['task'])

        all_observations['decision'].extend(observations['response'])
        all_observations['reaction_time'].extend(observations['rt'])

    all_observations = pd.DataFrame(all_observations)

    # replace nan with max rt in reaction_time and random decision

    all_observations['reaction_time'] = all_observations['reaction_time'].fillna(2000)
    all_observations['decision'] = all_observations['decision'].fillna(
        lambda _: random.choice([0, 1])
    )

    all_observations['decision'] = all_observations['decision'].astype('category')

    return Delta(models=[theorist.fit(all_conditions, all_observations)])


# ** Experimentalist ** #
# Here, we use a random pool and use the wrapper to create an on state function
# Note: The argument num_samples is not a state field. Instead, we will pass it in when calling
# the function


@on_state()
def experimentalist_on_state(models, num_samples):
    conditions = {'trial_sequence': []}
    for _ in range(num_samples):
        if len(models < 2):
            congruency_freq = random.choice([25, 50, 75])

            color = Factor('color', ['red', 'green'])
            word = Factor('word', ['RED', 'GREEN'])

            congruency = Factor('congruency', [
                DerivedLevel('congruent',
                             WithinTrial(lambda x, y: x.lower() == y.lower(), [color, word]),
                             weight=int(congruency_freq / 25)),
                DerivedLevel('incongruent',
                             WithinTrial(lambda x, y: x.lower() != y.lower(), [color, word]),
                             weight=int(4 - congruency_freq / 25)),
            ])

            design = [color, word, congruency]
            crossing = [color, congruency]
            constraints = []

            cross_block = CrossBlock(design, crossing, constraints)
            _timelines = synthesize_trials(cross_block, 1)
            timelines = experiments_to_dicts(cross_block, _timelines)
            conditions['trial_sequence'].append(timelines[0])
        else:
            scored_conditions = []
            for _ in range(100):
                congruency_freq = random.choice([25, 50, 75])

                color = Factor('color', ['red', 'green'])
                word = Factor('word', ['RED', 'GREEN'])
                task = Factor('task', ['word reading', 'color naming'])

                congruency = Factor('congruency', [
                    DerivedLevel('congruent',
                                 WithinTrial(lambda x, y: x.lower() == y.lower(), [color, word]),
                                 weight=int(congruency_freq / 25)),
                    DerivedLevel('incongruent',
                                 WithinTrial(lambda x, y: x.lower() != y.lower(), [color, word]),
                                 weight=int(4 - congruency_freq / 25)),
                ])

                design = [color, word, task, congruency]
                crossing = [color, task, congruency]
                constraints = []

                cross_block = CrossBlock(design, crossing, constraints)
                _timelines = synthesize_trials(cross_block, 1)
                timelines = experiments_to_dicts(cross_block, _timelines)

                print(timelines[0])

                theorist = Stroop()

                all_conditions = {
                    theorist.color: [],
                    theorist.word: [],
                    theorist.task: [],
                }

                map_trial_color = lambda x: [1, 0] if x == 'red' else [0, 1]
                map_trial_word = lambda x: [1, 0] if x == 'RED' else [0, 1]
                map_trial_task = lambda x: [1, 0] if x == 'color naming' else [0, 1]

                for trial in timelines[0]:
                    all_conditions[theorist.color].append(map_trial_color(trial))
                    all_conditions[theorist.word].append(map_trial_word(trial))
                    all_conditions[theorist.task].append(map_trial_task(trial['task']))

                print(theorist.predict(all_conditions))
                pred_1 = models[-1].predict(all_conditions)
                pred_2 = models[-2].predict(all_conditions)
                dist = np.linalg.norm(pred_1 - pred_2)
                scored_conditions.append({
                    'score': 1/dist,
                    'timeline': timelines[0],
                })
            # sort by score and take the best num_samples
            scored_conditions = sorted(scored_conditions, key=lambda x: x['score'], reverse=True)
            for cond in scored_conditions[:num_samples]:
                conditions['trial_sequence'].append(cond['timeline'])
    return Delta(conditions=pd.DataFrame(conditions))


# ** Experiment Runner ** #
# We will run our experiment on firebase and need credentials. You will find them here:
# (https://console.firebase.google.com/)
#   -> project -> project settings -> service accounts -> generate new private key

with open('firebase_credentials.json') as f:
    firebase_credentials = json.load(f)

# simple experiment runner that runs the experiment on firebase
experiment_runner = firebase_runner(
    firebase_credentials=firebase_credentials,
    time_out=30,
    sleep_time=5)


# Again, we need to wrap the runner to use it on the state. Here, we send the raw conditions.
@on_state()
def runner_on_state(conditions):
    experiments = []
    # for _, row in conditions.iterrows():
    #     timelines = row['trial_sequence']
    #
    #
    #     seq = [
    #         Fixation(duration=800),
    #         Text(duration=2000,
    #              text=TimelineVariable('word'),
    #              color=TimelineVariable('color'),
    #              correct_key=FunctionVariable(
    #                  name='correct_key',
    #                  fct=lambda x: 'f' if x == 'red' else 'j',
    #                  args=[TimelineVariable('color')]),
    #              choices=['f', 'j']),
    #         Feedback(duration=800)
    #     ]
    #
    #     for timeline in timelines:
    #         block = Block(seq, timeline)
    #         experiment = Experiment([block]).to_js_string()
    #         experiments.append(experiment)
    #
    # conditions_to_send = conditions.copy()
    # conditions_to_send['experiment_code'] = experiments
    # data = experiment_runner(conditions_to_send)
    # data = process_autora(data, len(seq))
    # df = pd.DataFrame(data)

    # TODO: Implement your data processing here (raw data -> condition, observation pairs)
    df = pd.read_csv('sample.csv')

    df['congruent'] = df['bean_text.1'].str.lower() == df['bean_color.1'].str.lower()

    grouped = df.groupby('exp_id')
    conditions = []
    observations = []
    for name, group in grouped:
        i = group
        colors = group['bean_color.1'].tolist()
        words = group['bean_text.1'].tolist()
        response = group['response.1'].tolist()
        task = ['color naming'] * len(colors)
        rt = group['rt.1'].tolist()

        colors = [[1, 0] if c == 'red' else [0, 1] for c in colors]
        words = [[1, 0] if w == 'RED' else [0, 1] for w in words]
        task = [[0, 1] if t == 'color naming' else [1, 0] for t in task]
        response = [1 if r == 'f' else 0 for r in response]
        conditions.append({
            'color': colors,
            'word': words,
            'task': task,
        })

        observations.append({
            'response': response,
            'rt': rt,
        })

        # print(colors, words, correct)
        # print(group[['bean_text.1', 'bean_color.1', 'bean_correct.1', 'congruent']])
        # print()


    # Fill this with ivs and dvs
    summary = pd.DataFrame({
        'congruency_freq': conditions,
        'congruency_effect': observations,
    })

    return Delta(experiment_data=summary)


def report_linear_fit(model: LinearRegression, precision=4):
    coefs = model.coef_.flatten()
    intercept = np.round(model.intercept_, precision)[0]
    terms = []

    for i, coef in enumerate(coefs):
        name = variables.independent_variables[i].name
        rounded_coef = np.round(coef, precision)
        terms.append(f"{rounded_coef}·{name}")

    formula = " + ".join(terms)
    sign = "+" if intercept >= 0 else "-"
    return f"{variables.dependent_variables[0].name} = {formula} {sign} {abs(intercept)}"


# Now, we can run our components
for cycle in range(3):
    print(f'Starting cycle {cycle}')
    state = experimentalist_on_state(state, num_samples=2)  # Collect 2 conditions per iteration
    state = runner_on_state(state)
    state = theorist_on_state(state)
    print(state.experiment_data)
    state.experiment_data.to_csv(f'experiment_data_{cycle}.csv')

    print()
    print('*' * 20)
    print(f'Model in cycle {cycle}:')
    print(report_linear_fit(state.models[-1]))
    print('*' * 20)
    print()
