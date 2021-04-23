# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.Qvalues = {}
        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE **"

        Q = self.Qvalues

        if (state,action) not in Q.keys():
            Q[(state,action)] = 0
        return Q[(state,action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions=self.getLegalActions(state)
        mxValue = None
        for action in actions:
            value = self.getQValue(state,action)
            if mxValue == None or value > mxValue:
                mxValue = value
        if mxValue == None:
            mxValue = 0.0
        return mxValue


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        act = None
        mxQ = None
        Q = self.Qvalues
        for action in actions:
            Qv = self.getQValue(state,action)
            if act == None or Qv>mxQ:
                mxQ = Qv
                act = action
        return act


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        e = self.epsilon
        legalActions = self.getLegalActions(state)
        isRand = util.flipCoin(e)
        action = None

        if len(legalActions) == 0:
            return None

        if isRand:
            action = random.choice(legalActions)
        else:
            mxQ = None
            for legalAction in legalActions:
                Qv = self.getQValue(state,legalAction)
                if mxQ == None or Qv>mxQ:
                    mxQ = Qv
                    action =  legalAction
        return action



    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

      #  print("update ",state, action, nextState, reward)


        Q = self.Qvalues
        a = self.alpha
        y = self.discount

        actions = self.getLegalActions(nextState)
        mxQ = None
        for act in actions:
            Qv = self.getQValue(nextState,act)
            if mxQ == None or Qv > mxQ:
                mxQ = Qv
        if mxQ ==None:
            mxQ = 0
        if (state,action) not in Q.keys():
            Q[(state,action)]=0
        Q[(state,action)] = (1-a)*Q[(state,action)]+a*(reward+y*mxQ)
      #  print ("update ",mxQ,y,reward,a,Q[(state,action)])



    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()


    def getWeights(self):
        return self.weights


    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        featureVector = self.featExtractor.getFeatures(state,action)
        w= self.weights
        Q = 0
        for i in featureVector:
            Q+=featureVector[i]*w[i]
        return Q


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        Q = self.Qvalues
        a = self.alpha
        y = self.discount
        w = self.weights

        actions = self.getLegalActions(nextState)
        mxQ = None
        for act in actions:
            Qv = self.getQValue(nextState,act)
            if mxQ == None or Qv > mxQ:
                mxQ = Qv
        if mxQ ==None:
            mxQ = 0
        diff = (reward+y*mxQ)-self.getQValue(state,action)
        featureVector = self.featExtractor.getFeatures(state,action)
        for i in featureVector:
            w[i]=w[i]+a*diff*featureVector[i]


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
