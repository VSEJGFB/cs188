# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #print("testinfo")
        #print(successorGameState)
        #print(newPos)
        #print(newFood)
        foodList=newFood.asList()
        if len(foodList)==0:
            return 1000000000
        x,y=newPos
        mn=1000
        for ghost in newGhostStates:
            gx,gy= ghost.configuration.getPosition()
            if abs(gx-x)+abs(gy-y)<=1:
                return -1

        for food in foodList:
            fx,fy=food
            mn=min(mn,abs(fx-x)+abs(fy-y))

        score=1000000.0/len(foodList)+10.0/mn

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def dfs(self,gameState,agentIndex,depth):
        #print("depth=",depth)
        #print("numAgents=",gameState.getNumAgents())
        #print("self.depth=",self.depth)
        actions=gameState.getLegalActions(agentIndex)
        if len(actions)==0 or depth==1+gameState.getNumAgents()*self.depth:
            #print("depth=%s self.depth=%s numAgents=%s" % (depth,self.depth,gameState.getNumAgents()))
            #print("evaluation=",self.evaluationFunction(gameState))
            return (0,self.evaluationFunction(gameState))

        score=None
        optact=None
        suc=[]
        for action in actions:
            state=gameState.generateSuccessor(agentIndex,action)
            opt = self.dfs(state,(agentIndex+1)%gameState.getNumAgents(),depth+1)
             
            act,sucScore=opt
            #print("sucScore=",sucScore);
            suc.append(sucScore)

            if score == None:
                score = sucScore           
                optact=action
            else:
                if agentIndex == 0:
                    if sucScore > score:
                        score=sucScore
                        optact=action
                else:
                    if sucScore < score:
                        score=sucScore
                        optact=action
       # print("agent=",agentIndex)
       # print("depth=",depth)
       # print("optact=",optact)
       # print("sucScore=",suc) 
        return (optact,score)
            

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #print("start,numAgent=%s,depth=%s" % (gameState.getNumAgents(),self.depth))
        act,score=self.dfs(gameState,0,1)
        #print("getscore=",score)
        return act

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def dfs(self,gameState,agentIndex,depth,alpha,beta):
        actions=gameState.getLegalActions(agentIndex)
        if len(actions)==0 or depth==1+gameState.getNumAgents()*self.depth:
            return (0,self.evaluationFunction(gameState))
        score=None
        optact=None
        suc=[]
        for action in actions:
            state=gameState.generateSuccessor(agentIndex,action)
            opt = self.dfs(state,(agentIndex+1)%gameState.getNumAgents(),depth+1,alpha,beta)
            act,sucScore=opt
            suc.append(sucScore)
            if score == None:
                score = sucScore           
                optact=action
            else:
                if agentIndex == 0:
                    if sucScore > score:
                        score=sucScore
                        optact=action
                else:
                    if sucScore < score:
                        score=sucScore
                        optact=action
            if agentIndex == 0:
                alpha = max(alpha , score)
            else:
                beta = min(beta , score)
            if alpha > beta:
                break
        return (optact,score)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        opt,score=self.dfs(gameState,0,1,-2e9,2e9)
        print("start,numAgent=%s,depth=%s" % (gameState.getNumAgents(),self.depth))
        print("getscore=",score)
        return opt


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def dfs(self,gameState,agentIndex,depth):
        actions=gameState.getLegalActions(agentIndex)
        if len(actions)==0 or depth==1+gameState.getNumAgents()*self.depth:
            return (0,self.evaluationFunction(gameState))
        score=None
        totscore=0
        optact=None
        suc=[]
        for action in actions:
            state=gameState.generateSuccessor(agentIndex,action)
            opt = self.dfs(state,(agentIndex+1)%gameState.getNumAgents(),depth+1)
            act,sucScore=opt
            suc.append(sucScore)
            if score == None:
                score = sucScore           
                optact=action
            else:
                if agentIndex == 0:
                    if sucScore > score:
                        score=sucScore
                        optact=action
                else:
                    if sucScore < score:
                        score=sucScore
                        optact=action
            totscore+=sucScore
        if agentIndex == 0:
            return (optact,score)
        else:
            return (optact,1.0*totscore/len(actions))


    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        opt,score=self.dfs(gameState,0,1)
        #print("start,numAgent=%s,depth=%s" % (gameState.getNumAgents(),self.depth))
        #print("getscore=",score)
        return opt

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    foodList=newFood.asList()
    if len(foodList)==0:
        return 2000000000
    x,y=newPos
    mn=1000
    for ghost in newGhostStates:
        gx,gy= ghost.configuration.getPosition()
        if abs(gx-x)+abs(gy-y)<=1:
            return -10000000

    for food in foodList:
        fx,fy=food
        mn=min(mn,abs(fx-x)+abs(fy-y))

    #score=12
    score=-100*len(foodList)+(-1)*mn
    print("score=",currentGameState.getScore())
    return score
    """

    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    walls = currentGameState.getWalls()

    # if not new ScaredTimes new state is ghost: return lowest value

    newFood = newFood.asList()
    ghostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
    scared = min(newScaredTimes) > 0


    if currentGameState.isLose():
        return float('-inf')

    if newPos in ghostPos:
        return float('-inf')


    # if not new ScaredTimes new state is ghost: return lowest value

    closestFoodDist = sorted(newFood, key=lambda fDist: util.manhattanDistance(fDist, newPos))
    closestGhostDist = sorted(ghostPos, key=lambda gDist: util.manhattanDistance(gDist, newPos))

    score = 0

    fd = lambda fDis: util.manhattanDistance(fDis, newPos)
    gd = lambda gDis: util.manhattanDistance(gDis, newPos)

    if gd(closestGhostDist[0]) <3:
        score-=300
    if gd(closestGhostDist[0]) <2:
        score-=1000
    if gd(closestGhostDist[0]) <1:
        return float('-inf')

    if len(currentGameState.getCapsules()) < 2:
        score+=100

    if len(closestFoodDist)==0 or len(closestGhostDist)==0 :
        score += scoreEvaluationFunction(currentGameState) + 10
    else:
        score += (   scoreEvaluationFunction(currentGameState) + 10/fd(closestFoodDist[0]) + 1/gd(closestGhostDist[0]) + 1/gd(closestGhostDist[-1])  )

    return score

# Abbreviation
better = betterEvaluationFunction
