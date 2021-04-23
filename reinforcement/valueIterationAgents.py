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

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        values = [util.Counter(), util.Counter()]
        states = self.mdp.getStates()
        for i in range(0, self.iterations):
            j = i & 1
            curValues = values[j]
            preValues = values[j ^ 1]
            self.values = preValues
            for state in states:
                curValues[state] = None
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    qValue = self.computeQValueFromValues(state, action)
                    if curValues[state]==None or qValue > curValues[state]:
                        curValues[state] = qValue
                if curValues[state]==None:
                    curValues[state]=0
        self.values = values[self.iterations & 1 ^ 1]



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

        qValue = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for transition in transitions:
            nextState, prob = transition
            reward = self.mdp.getReward(state, action, nextState)


            qValue += prob * (reward + self.discount * self.values[nextState])
        return qValue
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
        actions = self.mdp.getPossibleActions(state)
        maxValue = -2e9
        argMax = None
        for action in actions:
            qValue = self.computeQValueFromValues(state, action)
            if qValue > maxValue:
                maxValue = qValue
                argMax = action
        return argMax
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

    def __init__(self, mdp, discount=0.9, iterations=1000):
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
    #    print("iteraton=",self.iterations)
        states = self.mdp.getStates()

        i = int(1)
        while True:
            for state in states:
                mxQValue = None
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    qValue = self.computeQValueFromValues(state, action)
                    if mxQValue==None or qValue > mxQValue:
                        mxQValue = qValue


        #        print(mxQValue)


                if mxQValue==None:
                    self.values[state]=0
                else:
                    self.values[state]=mxQValue
                i += 1
                if i > self.iterations:
                    break
            if i > self.iterations:
                break
"""
        print("value:")
        for state in states:
            print(state,self.values[state])
        print("wtf")
"""
class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        from collections import defaultdict
        predecessors= defaultdict(set)
        states = self.mdp.getStates()
        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for transition in transitions:
                    nextState, prob = transition
                    if prob != 0:
                        predecessors[nextState].add(state)
        from util import PriorityQueue
        q = PriorityQueue()

        for state in states:

            if self.mdp.isTerminal(state):
                continue

            mxQValue = None
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                qValue = self.computeQValueFromValues(state, action)
                if mxQValue == None or qValue > mxQValue:
                    mxQValue = qValue
            if mxQValue == None:
                mxQValue = 0
         #   print("push ",state, -mxQValue)
            q.update(state, - abs(self.values[state]-mxQValue))

        for i in range(0,self.iterations):

            if q.isEmpty():
                break
            state = q.pop()

        #    print("state=",state)


            if not self.mdp.isTerminal(state):
                mxQValue = None
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    qValue = self.computeQValueFromValues(state, action)
                    if mxQValue == None or qValue > mxQValue:
                        mxQValue = qValue
                if mxQValue == None:
                    self.values[state]=0
                else:
                    self.values[state]=mxQValue
            for predecessor in predecessors[state]:
         #       print("pre=",predecessor)
                mxQValue = None
                actions = self.mdp.getPossibleActions(predecessor)
                for action in actions:
                    qValue = self.computeQValueFromValues(predecessor, action)
                    if mxQValue == None or qValue > mxQValue:
                        mxQValue = qValue
                if mxQValue == None:
                    mxQValue = 0
                if abs(mxQValue-self.values[predecessor]) > self.theta:
                   # print("update  ", predecessor, -mxQValue)
                    q.update(predecessor, - abs(mxQValue-self.values[predecessor]))
            """
            print("value:")
            for state in states:
                print(state, self.values[state])
            print("wtf")
            """