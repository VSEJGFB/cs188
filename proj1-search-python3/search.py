# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def adfs(cstate,prob,path,vis):
    
    if cstate in vis:
        return False
    vis.add(cstate)
    if prob.isGoalState(cstate):
        return True
    sucs=reversed(prob.getSuccessors(cstate))
    #sucs=prob.getSuccessors(cstate)
    for suc in sucs:
        if adfs(suc[0],prob,path,vis):
            path.append(suc[1])
            return True
    return False
  


def dffs(a,b,pro,p,v):
   print(a,b) 

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
 

   """
    "*** YOUR CODE HERE ***"

    path = []
    vis = set()
    adfs(problem.getStartState(),problem,path,vis)
    path.reverse()
    return path
  

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    startState = problem.getStartState()
    path = []
    q = Queue()
    q.push(startState)
    s = set()
    s.add(startState)
    pre = {}
    while not q.isEmpty():
        state = q.pop()
        if problem.isGoalState(state):
            break
        for suc in problem.getSuccessors(state):
            nxState=suc[0];
            if nxState not in s:
                s.add(nxState) 
                pre[nxState]=(state, suc[1])
                q.push(nxState)
    while state!=startState:
        path.append(pre[state][1])
        state=pre[state][0]
    path.reverse()
    return path

        

    

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue
    startState = problem.getStartState()
    path = []
    s = set()
    s.add(startState)
    pre = {}
    val = {}
    val[startState] = 0
    q = PriorityQueue()
    q.push(startState,0)
    while not q.isEmpty():
        state = q.pop()
        if problem.isGoalState(state):
            break
        for suc in problem.getSuccessors(state):
            nxState=suc[0];
            if nxState not in s:
                s.add(nxState) 
                pre[nxState]=(state, suc[1])
                val[nxState] = val[state]+suc[2]
                q.push(nxState,val[nxState])
            elif val[state]+suc[2]<val[nxState]:
                pre[nxState]=(state, suc[1])
                val[nxState]=val[state]+suc[2]
                q.update(nxState,val[nxState])
    while state!=startState:
        path.append(pre[state][1])
        state=pre[state][0]
    path.reverse()
    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    startState = problem.getStartState()
    path = []
    s = set()
    s.add(startState)
    pre = {}
    val = {}
    val[startState] = 0
    q = PriorityQueue()
    q.push(startState,0)
    while not q.isEmpty():
        state = q.pop()
        if problem.isGoalState(state):
            goalState=state
            break
        for suc in problem.getSuccessors(state):
            nxState=suc[0];
            if nxState not in s:
                s.add(nxState) 
                pre[nxState]=(state, suc[1])
                val[nxState] = val[state]+suc[2]
                q.push(nxState,val[nxState]+heuristic(nxState,problem))
            elif val[state]+suc[2]<val[nxState]:
                pre[nxState]=(state, suc[1])
                val[nxState]=val[state]+suc[2]
                q.update(nxState,val[nxState]+heuristic(nxState,problem))

    state=goalState
    while state!=startState:
        path.append(pre[state][1])
        state=pre[state][0]
    path.reverse()
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
