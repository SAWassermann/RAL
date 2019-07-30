import random
from copy import deepcopy

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class TrainingData:
    def __init__(self, X, y):
        self._X = deepcopy(X)
        self._y = deepcopy(y)

    def append(self, x, y):
        self._X.append(x)
        self._y.append(y)

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y


class RAL:
    def __init__(self, committee_classifiers, threshold_uncertainty, threshold_greedy, eta, reward=1, penalty=-1,
                 budget=None, model_update_frequency=1):
        self._committee_classifiers = deepcopy(committee_classifiers)
        self._threshold_uncertainty = threshold_uncertainty
        self._threshold_greedy = threshold_greedy
        self._eta = eta
        self._reward = reward
        self._penalty = penalty
        self._budget = budget
        self._model_update_frequency = model_update_frequency

        self._committee_classifier = VotingClassifier(
            list(zip(map(str, range(len(committee_classifiers))), self._committee_classifiers)), voting='soft')

        # Start with the same weight for each classifier
        self._alpha = [1.0 / len(self._committee_classifiers) for _ in range(len(self._committee_classifiers))]

        self._state = False  # Whether the committee predictors have been trained
        self._n_acquired_samples = 0  # Number of samples that were fed into RAL after the initial fit
        self._training_data_samples = None

        # Keep track of the algorithm's behaviour
        self._n_committee_too_low_acquires = 0  # Number of samples that the committee would have queried but
        # had no budget left
        self._n_committee_acquires = 0  # Number of positive decisions from the committee
        self._n_greedy_acquires = 0  # Number of greedy decisions
        self._n_both_acquires = 0  # Number of times where both the committee and greedy agreed to acquire
        self._n_seen = 0  # Number of times RAL saw a sample from the stream
        self._alphas = [deepcopy(self._alpha)]  # All values of alpha that this algorithm has been through

    @property
    def alphas(self):
        return self._alphas

    @property
    def n_committee_too_low_acquires(self):
        return self._n_committee_too_low_acquires

    @property
    def n_greedy_acquires(self):
        return self._n_greedy_acquires

    @property
    def n_committee_acquires(self):
        return self._n_committee_acquires

    @property
    def n_both_acquires(self):
        return self._n_both_acquires

    def fit(self, X, y):
        if self._state:
            raise AssertionError('fit has already been called.')

        self._training_data_samples = TrainingData(list(X), list(y))
        for learner in self._committee_classifiers:
            learner.fit(X, y)
        self._committee_classifier.fit(X, y)

        self._state = True

    def reward(self, y_true, y_predicted):
        if y_true != y_predicted:  # It was necessary to ask and committee took the right decision
            return self._reward
        else:  # It was not necessary to ask but committee decided to bother the user
            return self._penalty

    def _update_alpha(self, reward, decisions, committee_decision):
        # EXP4 algorithm
        for idx, decision in enumerate(decisions):
            # Only update the expert that has the same decision as the committee
            if int(decision) == int(committee_decision):
                self._alpha[idx] = self._alpha[idx] * np.exp(reward * self._eta)

        # Normalise the weights so they sum up to 1
        s = sum(self._alpha)
        for idx, coef in enumerate(self._alpha):
            self._alpha[idx] = float(coef) / s

        self._alphas.append(deepcopy(self._alpha))

    def _update_uncertainty_threshold(self, reward):
        f_reward = self._eta * (1. - np.power(2.0, float(reward) / self._penalty))
        self._threshold_uncertainty = min(self._threshold_uncertainty * (1. + f_reward), 1.)

    def _has_enough_budget(self):
        if self._budget is None:
            return True
        return float(self._n_acquired_samples) / self._n_seen < self._budget

    def ask_committee(self, x):
        # Gather the decisions of the committee
        decisions = []
        for learner in self._committee_classifiers:
            decisions.append(np.max(learner.predict_proba(x.reshape(1, -1))) < self._threshold_uncertainty)
        # If 50/50, do not ask
        committee_decision = bool(round(sum([self._alpha[idx] * el for idx, el in enumerate(decisions)])))
        return committee_decision, decisions

    def should_label(self, committee_decision):
        self._n_seen += 1

        if not self._has_enough_budget():
            if committee_decision:
                self._n_committee_too_low_acquires += 1
            return False

        if committee_decision:
            self._n_committee_acquires += 1

        # Implement an epsilon-greedy approach: with some probability, always ask, whatever the committee says.
        if random.random() < self._threshold_greedy:
            self._n_greedy_acquires += 1
            if committee_decision:
                self._n_both_acquires += 1

            return True
        else:
            return committee_decision

    def acquire_label(self, x, y, committee_decision, decisions):
        """Update RAL with a new sample"""

        # Update the state of RAL
        if committee_decision:
            yp = self._committee_classifier.predict(x.reshape(1, -1))[0]
            reward = self.reward(y, yp)
            self._update_alpha(reward, decisions, committee_decision)
            self._update_uncertainty_threshold(reward)

        # Keep track of the new data point
        self._n_acquired_samples += 1
        self._training_data_samples.append(x, y)

        # Every so often, update the models
        if self._n_acquired_samples % self._model_update_frequency == 0:
            for learner in self._committee_classifiers:
                learner.fit(self._training_data_samples.X, self._training_data_samples.y)
            self._committee_classifier.fit(self._training_data_samples.X, self._training_data_samples.y)

    def evaluate_performance(self, X, y):
        self._committee_classifier.fit(self._training_data_samples.X, self._training_data_samples.y)

        y_predicted = self._committee_classifier.predict(X)
        return accuracy_score(y, y_predicted)


if __name__ == '__main__':
    # Parameters
    THRESHOLD_GREEDY = 0.025
    THRESHOLD_UNCERTAINTY = .9
    ETA = 0.01
    REWARD = 1
    PENALTY = -1
    BUDGET = 0.05

    TRAIN_PORTION = 0.005
    VALIDATION_PORTION = 0.3
    STREAMING_PORTION = 1 - TRAIN_PORTION - VALIDATION_PORTION

    # Load here your data
    data_X = np.array([])
    data_y = np.array([])

    # Split the data set into 3 parts
    indices = list(range(len(data_y)))
    indices_train = indices[:int(round(TRAIN_PORTION * len(data_y)))]
    indices_stream = indices[len(indices_train):len(indices_train) + int(round(STREAMING_PORTION * len(data_y)))]
    indices_validation = indices[len(indices_train) + len(indices_stream):]

    X_base, y_base = data_X[indices_train], data_y[indices_train]
    X_stream, y_stream = data_X[indices_stream], data_y[indices_stream]
    X_valid, y_valid = data_X[indices_validation], data_y[indices_validation]

    # Committee
    knn = KNeighborsClassifier()
    rf10 = RandomForestClassifier()
    dt = DecisionTreeClassifier()

    learners = [knn, rf10, dt]  # Committee
    # learners = [rf10]  # Single classifier

    # RAL
    ral = RAL(learners, THRESHOLD_UNCERTAINTY, THRESHOLD_GREEDY, ETA, reward=REWARD, penalty=PENALTY, budget=BUDGET,
              model_update_frequency=1)
    ral.fit(X_base, y_base)
    init_acc = ral.evaluate_performance(X_valid, y_valid)

    for t, x_t in enumerate(X_stream):
        # Ask the committee for its opinion
        committeeDecision, learnerDecisions = ral.ask_committee(x_t)
        finalDecision = ral.should_label(committeeDecision)
        if finalDecision:
            ral.acquire_label(x_t, y_stream[t], committeeDecision, learnerDecisions)

    final_acc = ral.evaluate_performance(X_valid, y_valid)
