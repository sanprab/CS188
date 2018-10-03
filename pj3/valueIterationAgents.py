# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"


        for i in range(self.iterations): # every k
            updatedValues = self.values.copy()  # to use batch-version of MDP , hard copy the values

            for state in self.mdp.getStates():

                if self.mdp.isTerminal(state):
                    continue

                actions = self.mdp.getPossibleActions(state)
                optimal = max([self.getQValue(state,action) for action in actions])
                updatedValues[state] = optimal

            self.values = updatedValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        "*** YOUR CODE HERE ***"

        qval = 0

        for s_prime, T in self.mdp.getTransitionStatesAndProbs(state, action):
            qval += T * ( self.mdp.getReward(state, action, s_prime) + self.discount*self.getValue(s_prime) )

        return qval

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # TODO

        policy = util.Counter()

        for action in self.mdp.getPossibleActions(state):
            policy[action] = self.getQValue(state, action)

        return policy.argMax()

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        #TODO

        totalState = self.mdp.getStates()


        for i in range(self.iterations): # every k

            state = totalState[i % len(totalState)]

            if self.mdp.isTerminal(state):
                continue

            actions = self.mdp.getPossibleActions(state)
            optimal = max([self.getQValue(state,action) for action in actions])
            self.values[state] = optimal




class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        q = util.PriorityQueue()

        totalState = self.mdp.getStates()
        pred = {}

        for st in totalState:
            if self.mdp.isTerminal(st):
                continue
            for ac in self.mdp.getPossibleActions(st):
                for stt,_ in self.mdp.getTransitionStatesAndProbs(st, ac):
                    if stt in pred:
                        pred[stt].add(st)
                    else:
                        pred[stt] = {st}


        for st in self.mdp.getStates():
            if self.mdp.isTerminal(st):
                continue

            diff = abs(self.values[st] - max([ self.computeQValueFromValues(st, action) for action in self.mdp.getPossibleActions(st) ]) )

            q.update(st, -diff)

        for i in range(self.iterations):
            if q.isEmpty():
                break
            st = q.pop()
            if not self.mdp.isTerminal(st):
                self.values[st] = max([self.computeQValueFromValues(st, action) for action in self.mdp.getPossibleActions(st)])

            for p in pred[st]:

                if self.mdp.isTerminal(p):
                    continue

                difff = abs(self.values[p] - max([self.computeQValueFromValues(p, action) for action in self.mdp.getPossibleActions(p)]))

                if difff > self.theta:
                        q.update(p, -difff)
